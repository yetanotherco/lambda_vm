//! ZisK's Poseidon1 on the device must produce the host's bytes (`p1/*`
//! exploration branch, P2): every `DeviceHash::Poseidon1` leaf geometry, the
//! 4-ary trees, the path gather, the kept-top prefix and the grind, against
//! the production host configuration (`P1StarkHash`: `P1BatchBackend`,
//! `P1PairBackend`, `P1GrindDigest`), whose primitives are pinned to ZisK's own
//! vectors (`crypto::hash::poseidon1_stark::zisk_kat`). The kernels' arithmetic
//! and index math are also pinned without a GPU (`make test-p1w16-host-kat`,
//! against ZisK's trees and paths); what needs the GPU is here. The twin of
//! `rpx_device_parity.rs`. Needs a GPU:
//!
//!   cargo test -p lambda-vm-prover --release --features cuda --test p1_device_parity -- --nocapture
#![cfg(feature = "cuda")]

use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use lambda_vm_prover::lfm::p1_commit::{P1BatchBackend, P1PairBackend, P1StarkHash};
use math::fft::two_half_fft::TwoHalfTwiddles;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::polynomial::Polynomial;
use stark::prover::{GenericProver, IsStarkProver};

/// The host prover under ZisK's Poseidon1, named.
type Prover<F, E, PI> = GenericProver<F, E, PI, P1StarkHash>;

type Ext3 = Degree3GoldilocksExtensionField;
type Fp3 = FieldElement<Ext3>;
type Fp = FieldElement<GoldilocksField>;

const COSET_OFFSET: u64 = 7;

/// splitmix64.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

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

fn rand_ext3(rng: &mut SplitMix) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.next()),
        Fp::from_raw(rng.next()),
        Fp::from_raw(rng.next()),
    ])
}

fn nodes_of(bytes: &[u8]) -> Vec<[u8; 32]> {
    bytes
        .chunks_exact(32)
        .map(|c| c.try_into().expect("32-byte node"))
        .collect()
}

// ===========================================================================
// The fused LDE + commit pipelines (the trace trees): both leaf layouts, both
// LDE engines, even and odd 4-ary depths.
// ===========================================================================

/// The host LDE of `columns` (row-major out) and its P1 commit root at
/// `rows_per_leaf`.
fn cpu_root(columns: &[Vec<u64>], blowup: usize, rows_per_leaf: usize) -> [u8; 32] {
    let n = columns[0].len();
    let num_cols = columns.len();
    let log_n = n.trailing_zeros() as usize;
    let log_lde = (n * blowup).trailing_zeros() as usize;
    let mut buf: Vec<Fp> = vec![Fp::from(0u64); n * num_cols];
    for (c, col) in columns.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            buf[r * num_cols + c] = Fp::from_raw(v);
        }
    }
    let inv_tw = TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).expect("inv twiddles");
    let fwd_tw = TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).expect("fwd twiddles");
    Polynomial::<Fp>::coset_lde_full_expand_row_major::<GoldilocksField>(
        &mut buf,
        num_cols,
        blowup,
        &coset_weights(n, COSET_OFFSET),
        &inv_tw,
        &fwd_tw,
    )
    .expect("CPU row-major LDE");
    let (_, root) = Prover::<GoldilocksField, GoldilocksField, ()>::commit_rows_bit_reversed_with(
        &buf,
        num_cols,
        rows_per_leaf,
    )
    .expect("CPU P1 commit");
    root
}

#[test]
fn p1_fused_trace_roots_match_the_host() {
    for log_n in [4usize, 5, 7, 10] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 5, 12, 13] {
                for rows_per_leaf in [1usize, 2] {
                    // `retain_host_lde = true` takes the row-major leaf kernels,
                    // `false` the column-major engine's.
                    for retain in [true, false] {
                        let n = 1usize << log_n;
                        let mut rng =
                            SplitMix((log_n * 10_000 + blowup * 1000 + num_cols * 10) as u64);
                        let columns: Vec<Vec<u64>> = (0..num_cols)
                            .map(|_| (0..n).map(|_| rng.next()).collect())
                            .collect();
                        let mut row_major = vec![0u64; n * num_cols];
                        for (c, col) in columns.iter().enumerate() {
                            for (r, &v) in col.iter().enumerate() {
                                row_major[r * num_cols + c] = v;
                            }
                        }
                        let weights: Vec<u64> = coset_weights(n, COSET_OFFSET)
                            .iter()
                            .map(|w| *w.value())
                            .collect();
                        let (handle, _) =
                            math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep_rpl(
                                &row_major,
                                None,
                                math_cuda::DeviceHash::Poseidon1,
                                n,
                                num_cols,
                                blowup,
                                &weights,
                                retain,
                                rows_per_leaf,
                            )
                            .expect("fused P1 pipeline");
                        let tree = handle.tree.as_ref().expect("resident tree");
                        assert_eq!(tree.arity, 4);
                        assert_eq!(
                            tree.root,
                            cpu_root(&columns, blowup, rows_per_leaf),
                            "log_n={log_n} blowup={blowup} cols={num_cols} rpl={rows_per_leaf} \
                             retain={retain}"
                        );
                    }
                }
            }
        }
    }
}

/// The ext3 (aux) trace commit, row pairs.
#[test]
fn p1_fused_ext3_roots_match_the_host() {
    for log_n in [4usize, 7] {
        for num_cols in [1usize, 4] {
            for rows_per_leaf in [1usize, 2] {
                let n = 1usize << log_n;
                let blowup = 4usize;
                let mut rng = SplitMix((log_n * 100 + num_cols) as u64 + 77);
                let columns: Vec<Vec<Fp3>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
                    .collect();
                let mut row_major = vec![0u64; n * num_cols * 3];
                let mut buf: Vec<Fp3> = vec![Fp3::from(0u64); n * num_cols];
                for (c, col) in columns.iter().enumerate() {
                    for (r, v) in col.iter().enumerate() {
                        buf[r * num_cols + c] = *v;
                        for k in 0..3 {
                            row_major[(r * num_cols + c) * 3 + k] = *v.value()[k].value();
                        }
                    }
                }
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let weights: Vec<u64> = weights_fp.iter().map(|w| *w.value()).collect();
                let (handle, _) =
                    math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep_rpl(
                        &row_major,
                        math_cuda::DeviceHash::Poseidon1,
                        n,
                        num_cols,
                        blowup,
                        &weights,
                        true,
                        rows_per_leaf,
                    )
                    .expect("fused ext3 P1 pipeline");
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let inv_tw = TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).unwrap();
                let fwd_tw = TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).unwrap();
                Polynomial::<Fp3>::coset_lde_full_expand_row_major::<GoldilocksField>(
                    &mut buf,
                    num_cols,
                    blowup,
                    &weights_fp,
                    &inv_tw,
                    &fwd_tw,
                )
                .unwrap();
                let (_, want) = Prover::<GoldilocksField, Ext3, ()>::commit_rows_bit_reversed_with(
                    &buf,
                    num_cols,
                    rows_per_leaf,
                )
                .unwrap();
                assert_eq!(
                    handle.tree.as_ref().expect("resident tree").root,
                    want,
                    "ext3 log_n={log_n} cols={num_cols} rpl={rows_per_leaf}"
                );
            }
        }
    }
}

/// Column ranges (the preprocessed tables' two trees over one LDE), leaf for
/// leaf against the Batch backend's `hash_data` over the same felts.
#[test]
fn p1_row_major_range_leaves_match_the_host() {
    let log_n = 6u32;
    let n = 1usize << log_n;
    let m = 7usize;
    let mut rng = SplitMix(0xBEEF);
    let data: Vec<u64> = (0..n * m).map(|_| rng.next()).collect();
    let reverse_index = |i: usize| -> usize { (i as u64).reverse_bits() as usize >> (64 - log_n) };
    for rows_per_leaf in [1usize, 2] {
        for (cs, ce) in [(0usize, m), (0, 3), (3, m), (2, 5)] {
            let gpu = math_cuda::lde::row_major_leaves(
                math_cuda::DeviceHash::Poseidon1,
                &data,
                m,
                cs,
                ce,
                n,
                rows_per_leaf,
            )
            .expect("device ranged leaves");
            assert_eq!(gpu.len(), n / rows_per_leaf * 32);
            for leaf in 0..n / rows_per_leaf {
                let mut felts: Vec<Fp> = Vec::new();
                for k in 0..rows_per_leaf {
                    let br = reverse_index(rows_per_leaf * leaf + k);
                    for c in cs..ce {
                        felts.push(Fp::from_raw(data[br * m + c]));
                    }
                }
                let want =
                    <P1BatchBackend<GoldilocksField> as IsMerkleTreeBackend>::hash_data(&felts);
                assert_eq!(
                    &gpu[leaf * 32..(leaf + 1) * 32],
                    &want[..],
                    "leaf {leaf}, columns [{cs}, {ce}), rows_per_leaf {rows_per_leaf}"
                );
            }
        }
    }
}

// ===========================================================================
// The composition tree, the FRI trees and the bare 4-ary walk, node for node.
// ===========================================================================

#[test]
fn p1_comp_poly_trees_match_the_host() {
    for (log_lde, m) in [(4usize, 1usize), (5, 2), (10, 3), (13, 4)] {
        for rows_per_leaf in [1usize, 2] {
            let lde_size = 1usize << log_lde;
            let mut rng = SplitMix((log_lde * 10 + m) as u64 + 99);
            let parts: Vec<Vec<Fp3>> = (0..m)
                .map(|_| (0..lde_size).map(|_| rand_ext3(&mut rng)).collect())
                .collect();
            let parts_u64: Vec<Vec<u64>> = parts
                .iter()
                .map(|p| {
                    p.iter()
                        .flat_map(|e| e.value().iter().map(|c| *c.value()))
                        .collect()
                })
                .collect();
            let raw: Vec<&[u64]> = parts_u64.iter().map(|p| p.as_slice()).collect();
            let dev = math_cuda::p1_stark::build_comp_poly_tree_from_evals_ext3_keep_rpl(
                &raw,
                rows_per_leaf,
            )
            .expect("device comp-poly tree");
            assert_eq!(dev.leaves_len, lde_size / rows_per_leaf);
            let mut buf: Vec<Fp3> = vec![Fp3::from(0u64); lde_size * m];
            for (c, p) in parts.iter().enumerate() {
                for (r, v) in p.iter().enumerate() {
                    buf[r * m + c] = *v;
                }
            }
            let (host, want) = Prover::<GoldilocksField, Ext3, ()>::commit_rows_bit_reversed_with(
                &buf,
                m,
                rows_per_leaf,
            )
            .unwrap();
            assert_eq!(
                dev.root, want,
                "log_lde={log_lde} m={m} rpl={rows_per_leaf}"
            );

            // The whole node buffer, then paths and the kept-top prefix.
            let be = math_cuda::device::backend().expect("backend");
            let stream = be.next_stream();
            let all = math_cuda::lde::download_tree_prefix(&dev, usize::MAX).unwrap();
            assert_eq!(
                all.as_slice(),
                host.nodes(),
                "node buffer log_lde={log_lde}"
            );
            assert_eq!(all.len(), dev.total_nodes());
            let positions: Vec<u32> = [0usize, 1, 3, dev.leaves_len / 2, dev.leaves_len - 1]
                .iter()
                .map(|&p| p as u32)
                .collect();
            let bytes = math_cuda::p1_stark::gather_paths_dev(
                &dev.nodes,
                dev.leaves_len,
                &positions,
                &stream,
            )
            .expect("device paths");
            let per = 3 * math_cuda::p1_stark::depth(dev.leaves_len) * 32;
            for (q, &p) in positions.iter().enumerate() {
                let want_path = host
                    .get_proof_by_pos(p as usize)
                    .expect("host path")
                    .merkle_path;
                assert_eq!(
                    nodes_of(&bytes[q * per..(q + 1) * per]),
                    want_path,
                    "path {p} log_lde={log_lde} rpl={rows_per_leaf}"
                );
            }
        }
    }
}

fn fri_tree_parity(num_leaves: usize, group: usize, seed: u64) {
    let mut rng = SplitMix(seed);
    let evals: Vec<Fp3> = (0..num_leaves * group)
        .map(|_| rand_ext3(&mut rng))
        .collect();
    let evals_u64: Vec<u64> = evals
        .iter()
        .flat_map(|e| e.value().iter().map(|c| *c.value()))
        .collect();
    let host_nodes: Vec<[u8; 32]> = if group == 2 {
        let leaves: Vec<[Fp3; 2]> = evals.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        MerkleTree::<P1PairBackend<Ext3>>::build(&leaves)
            .unwrap()
            .nodes()
            .to_vec()
    } else {
        let leaves: Vec<Vec<Fp3>> = evals.chunks_exact(group).map(<[Fp3]>::to_vec).collect();
        MerkleTree::<P1BatchBackend<Ext3>>::build(&leaves)
            .unwrap()
            .nodes()
            .to_vec()
    };
    let gpu = math_cuda::p1_stark::build_fri_tree_from_evals_ext3(&evals_u64, group).unwrap();
    assert_eq!(
        nodes_of(&gpu),
        host_nodes,
        "FRI tree: {num_leaves} leaves of {group}"
    );
}

/// Pair and group FRI trees: shallow (the tail alone) and deep enough that the
/// per-level kernel runs first, at even and odd 4-ary depths.
#[test]
fn p1_fri_trees_match_the_host() {
    for log in [1u32, 2, 3, 5, 6, 11, 14, 15] {
        for group in [2usize, 4, 16] {
            fri_tree_parity(1 << log, group, 300 + log as u64 * 7 + group as u64);
        }
    }
}

/// The bare 4-ary walk over random leaf digests, node for node, from two
/// leaves (the tail alone) to 2^13 (per-level launches, then the tail).
#[test]
fn p1_bare_trees_match_the_host() {
    for leaves in [2usize, 4, 8, 32, 64, 1 << 10, 1 << 13] {
        let mut rng = SplitMix(leaves as u64);
        let hashed: Vec<[u8; 32]> = (0..leaves)
            .map(|_| {
                let d = [
                    Fp::from(rng.next()),
                    Fp::from(rng.next()),
                    Fp::from(rng.next()),
                    Fp::from(rng.next()),
                ];
                lambda_vm_prover::lfm::algebraic_commit::digest_to_commitment(&d)
            })
            .collect();
        let host =
            MerkleTree::<P1BatchBackend<GoldilocksField>>::build_from_hashed_leaves(hashed.clone())
                .unwrap();
        let flat: Vec<u8> = hashed.iter().flatten().copied().collect();
        let gpu = math_cuda::p1_stark::build_merkle_tree_on_device(&flat).unwrap();
        assert_eq!(nodes_of(&gpu), host.nodes(), "{leaves} leaves");
    }
}

// ===========================================================================
// The grind.
// ===========================================================================

type P1Grind = stark::config::GrindingDigest<P1StarkHash>;

/// A real launch returns the smallest nonce the host predicate accepts, at
/// factors from the device floor (`GRIND_MIN_FACTOR`) up; below it the device
/// declines (`None`) and the dispatcher's host search takes the grind.
#[test]
fn p1_gpu_grind_returns_the_smallest_valid_nonce() {
    let floor = math_cuda::grinding::GRIND_MIN_FACTOR;
    for (seed, factor) in [([14u8; 32], floor), ([3u8; 32], 14u8), ([77u8; 32], 16)] {
        let nonce = math_cuda::grinding::generate_nonce_p1_gpu(
            &stark::grinding::inner_hash_felts::<P1Grind>(&seed, factor),
            factor,
        )
        .unwrap_or_else(|| panic!("GPU P1 grind at factor {factor} (needs a GPU)"));
        assert!(
            stark::grinding::is_valid_nonce::<P1Grind>(&seed, nonce, factor),
            "GPU nonce {nonce} fails is_valid_nonce (factor {factor})"
        );
        assert!(
            (0..nonce).all(|n| !stark::grinding::is_valid_nonce::<P1Grind>(&seed, n, factor)),
            "GPU nonce {nonce} is not the smallest (factor {factor})"
        );
    }
    let below = floor - 2;
    assert!(
        math_cuda::grinding::generate_nonce_p1_gpu(
            &stark::grinding::inner_hash_felts::<P1Grind>(&[3u8; 32], below),
            below,
        )
        .is_none(),
        "the device grind must decline factor {below}, below its floor {floor}"
    );
}

/// The production dispatch reaches the Poseidon1 kernel: the P1 counter moves,
/// the RPX one does not.
#[test]
fn the_p1_grind_dispatch_reaches_the_device() {
    crypto::grinding::reset_gpu_grind_calls();
    let seed = [9u8; 32];
    let nonce = stark::grinding::generate_nonce_maybe_gpu::<P1Grind>(&seed, 18).expect("nonce");
    assert!(stark::grinding::is_valid_nonce::<P1Grind>(&seed, nonce, 18));
    assert_eq!(
        crypto::grinding::gpu_grind_calls_p1(),
        1,
        "the P1 device grind ran"
    );
    assert_eq!(crypto::grinding::gpu_grind_calls_rpx(), 0, "no RPX grind");
}
