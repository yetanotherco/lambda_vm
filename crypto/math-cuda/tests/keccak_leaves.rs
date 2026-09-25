//! Parity: GPU Keccak-256 leaf hashes must match the CPU prover's leaf
//! hashing helpers. `stark::prover::keccak_leaves_bit_reversed` for
//! per-row commits, `keccak_leaves_row_pair_bit_reversed` for the R2
//! composition commit, and `FriLayerMerkleTreeBackend::hash_data` for the
//! FRI commit. These are the same helpers the prover itself calls so any
//! change to the CPU leaf-hash contract surfaces here.

use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::config::KeccakFriLayerMerkleTreeBackend as FriLayerMerkleTreeBackend;
use stark::prover::{keccak_leaves_bit_reversed, keccak_leaves_row_pair_bit_reversed};

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

#[test]
fn keccak_leaves_base_matches_cpu() {
    for log_n in [4u32, 6, 8, 10, 12] {
        for num_cols in [1usize, 5, 17, 41] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(100 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fp::from_raw(rng.r#gen::<u64>())).collect())
                .collect();

            let cpu = keccak_leaves_bit_reversed(&columns);

            // Flatten columns into a contiguous base slab layout matching
            // `coset_lde_batch_base_into`'s pinned staging format:
            // `[col * stride + row]`. Use stride = num_rows for compactness.
            let mut flat = vec![0u64; num_cols * n];
            for (c, col) in columns.iter().enumerate() {
                for (r, e) in col.iter().enumerate() {
                    flat[c * n + r] = *e.value();
                }
            }
            let gpu = math_cuda::merkle::keccak_leaves_base(&flat, n, num_cols, n, 1).unwrap();
            assert_eq!(gpu.len(), n * 32);
            for i in 0..n {
                assert_eq!(
                    &gpu[i * 32..(i + 1) * 32],
                    &cpu[i][..],
                    "base leaf mismatch at row {i} (log_n={log_n}, cols={num_cols})"
                );
            }
        }
    }
}

#[test]
fn keccak_leaves_ext3_matches_cpu() {
    for log_n in [4u32, 6, 8, 10] {
        for num_cols in [1usize, 3, 11, 20] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(200 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp3>> = (0..num_cols)
                .map(|_| {
                    (0..n)
                        .map(|_| {
                            Fp3::new([
                                Fp::from_raw(rng.r#gen::<u64>()),
                                Fp::from_raw(rng.r#gen::<u64>()),
                                Fp::from_raw(rng.r#gen::<u64>()),
                            ])
                        })
                        .collect()
                })
                .collect();

            let cpu = keccak_leaves_bit_reversed(&columns);

            // GPU expects 3 base slabs per ext3 column in the order
            // [col*3+0 (comp a), col*3+1 (comp b), col*3+2 (comp c)], each a
            // contiguous slab of n u64s (length = num_cols * 3 * n).
            let mut flat = vec![0u64; num_cols * 3 * n];
            for (c, col) in columns.iter().enumerate() {
                for (r, e) in col.iter().enumerate() {
                    flat[(c * 3) * n + r] = *e.value()[0].value();
                    flat[(c * 3 + 1) * n + r] = *e.value()[1].value();
                    flat[(c * 3 + 2) * n + r] = *e.value()[2].value();
                }
            }
            let gpu = math_cuda::merkle::keccak_leaves_ext3(&flat, n, num_cols, n, 1).unwrap();
            assert_eq!(gpu.len(), n * 32);
            for i in 0..n {
                assert_eq!(
                    &gpu[i * 32..(i + 1) * 32],
                    &cpu[i][..],
                    "ext3 leaf mismatch at row {i} (log_n={log_n}, cols={num_cols})"
                );
            }
        }
    }
}

#[test]
fn keccak_leaves_base_row_pair_matches_cpu() {
    // Row-pair (trace) commit: leaf `i` hashes bit-reversed rows `2i`, `2i+1`.
    // GPU `keccak_leaves_base(.., rows_per_leaf=2)` must match the CPU prover
    // helper `keccak_leaves_row_pair_bit_reversed` over base columns.
    for log_n in [4u32, 6, 8, 10, 12] {
        for num_cols in [1usize, 5, 17, 41] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(500 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fp::from_raw(rng.r#gen::<u64>())).collect())
                .collect();

            let cpu = keccak_leaves_row_pair_bit_reversed(&columns);
            assert_eq!(cpu.len(), n / 2);

            let mut flat = vec![0u64; num_cols * n];
            for (c, col) in columns.iter().enumerate() {
                for (r, e) in col.iter().enumerate() {
                    flat[c * n + r] = *e.value();
                }
            }
            let gpu = math_cuda::merkle::keccak_leaves_base(&flat, n, num_cols, n, 2).unwrap();
            assert_eq!(gpu.len(), (n / 2) * 32);
            for i in 0..n / 2 {
                assert_eq!(
                    &gpu[i * 32..(i + 1) * 32],
                    &cpu[i][..],
                    "base row-pair leaf mismatch at i={i} (log_n={log_n}, cols={num_cols})"
                );
            }
        }
    }
}

#[test]
fn keccak_leaves_ext3_row_pair_matches_cpu() {
    for log_n in [4u32, 6, 8, 10] {
        for num_cols in [1usize, 3, 11, 20] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(600 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp3>> = (0..num_cols)
                .map(|_| {
                    (0..n)
                        .map(|_| {
                            Fp3::new([
                                Fp::from_raw(rng.r#gen::<u64>()),
                                Fp::from_raw(rng.r#gen::<u64>()),
                                Fp::from_raw(rng.r#gen::<u64>()),
                            ])
                        })
                        .collect()
                })
                .collect();

            let cpu = keccak_leaves_row_pair_bit_reversed(&columns);
            assert_eq!(cpu.len(), n / 2);

            // De-interleaved 3-slab layout per ext3 column (same as the 1-row
            // ext3 leaf path): [col*3+k] each a contiguous slab of n u64s.
            let mut flat = vec![0u64; num_cols * 3 * n];
            for (c, col) in columns.iter().enumerate() {
                for (r, e) in col.iter().enumerate() {
                    flat[(c * 3) * n + r] = *e.value()[0].value();
                    flat[(c * 3 + 1) * n + r] = *e.value()[1].value();
                    flat[(c * 3 + 2) * n + r] = *e.value()[2].value();
                }
            }
            let gpu = math_cuda::merkle::keccak_leaves_ext3(&flat, n, num_cols, n, 2).unwrap();
            assert_eq!(gpu.len(), (n / 2) * 32);
            for i in 0..n / 2 {
                assert_eq!(
                    &gpu[i * 32..(i + 1) * 32],
                    &cpu[i][..],
                    "ext3 row-pair leaf mismatch at i={i} (log_n={log_n}, cols={num_cols})"
                );
            }
        }
    }
}

#[test]
fn keccak_comp_poly_leaves_matches_cpu() {
    // Built tree's leaves live at byte offset `(num_leaves - 1) * 32` and
    // span `num_leaves * 32` bytes. Compare those to the CPU reference.
    for log_lde in [2u32, 4, 6, 8, 10, 12] {
        for num_parts in [1usize, 2, 5, 17] {
            let lde_size = 1usize << log_lde;
            let mut rng = ChaCha8Rng::seed_from_u64(300 + log_lde as u64 + num_parts as u64);
            let parts: Vec<Vec<Fp3>> = (0..num_parts)
                .map(|_| {
                    (0..lde_size)
                        .map(|_| {
                            Fp3::new([
                                Fp::from_raw(rng.r#gen::<u64>()),
                                Fp::from_raw(rng.r#gen::<u64>()),
                                Fp::from_raw(rng.r#gen::<u64>()),
                            ])
                        })
                        .collect()
                })
                .collect();
            let cpu = keccak_leaves_row_pair_bit_reversed(&parts);

            // Each part is passed as `[a0,a1,a2, b0,b1,b2, ...]` of length `3 * lde_size`.
            let parts_interleaved: Vec<Vec<u64>> = parts
                .iter()
                .map(|p| {
                    let mut v = vec![0u64; 3 * lde_size];
                    for (i, e) in p.iter().enumerate() {
                        v[i * 3] = *e.value()[0].value();
                        v[i * 3 + 1] = *e.value()[1].value();
                        v[i * 3 + 2] = *e.value()[2].value();
                    }
                    v
                })
                .collect();
            let parts_slices: Vec<&[u64]> =
                parts_interleaved.iter().map(|v| v.as_slice()).collect();

            // Exercise the production keep path, then read the resident nodes
            // back to host to check the leaf bytes.
            let tree = math_cuda::merkle::build_comp_poly_tree_from_evals_ext3_keep(&parts_slices)
                .unwrap();
            let be = math_cuda::device::backend().unwrap();
            let stream = be.next_stream();
            let nodes: Vec<u8> = stream.clone_dtoh(&*tree.nodes).unwrap();
            let num_leaves = lde_size / 2;
            let leaves_offset = (num_leaves - 1) * 32;
            for i in 0..num_leaves {
                assert_eq!(
                    &nodes[leaves_offset + i * 32..leaves_offset + (i + 1) * 32],
                    &cpu[i][..],
                    "comp-poly leaf mismatch at i={i} (log_lde={log_lde}, parts={num_parts})"
                );
            }
        }
    }
}

#[test]
fn keccak_fri_leaves_matches_cpu() {
    for log_lde in [2u32, 4, 6, 8, 10, 12] {
        let lde_size = 1usize << log_lde;
        let mut rng = ChaCha8Rng::seed_from_u64(400 + log_lde as u64);
        let evals: Vec<Fp3> = (0..lde_size)
            .map(|_| {
                Fp3::new([
                    Fp::from_raw(rng.r#gen::<u64>()),
                    Fp::from_raw(rng.r#gen::<u64>()),
                    Fp::from_raw(rng.r#gen::<u64>()),
                ])
            })
            .collect();

        // CPU reference: consecutive ext3 pairs hashed via the prover's
        // FRI-layer Merkle backend.
        let cpu: Vec<[u8; 32]> = evals
            .chunks_exact(2)
            .map(|c| {
                FriLayerMerkleTreeBackend::<Degree3GoldilocksExtensionField>::hash_data(&[
                    c[0], c[1],
                ])
            })
            .collect();

        let mut evals_interleaved = vec![0u64; 3 * lde_size];
        for (i, e) in evals.iter().enumerate() {
            evals_interleaved[i * 3] = *e.value()[0].value();
            evals_interleaved[i * 3 + 1] = *e.value()[1].value();
            evals_interleaved[i * 3 + 2] = *e.value()[2].value();
        }
        let nodes =
            math_cuda::merkle::build_fri_layer_tree_from_evals_ext3(&evals_interleaved).unwrap();
        let num_leaves = lde_size / 2;
        let leaves_offset = (num_leaves - 1) * 32;
        for i in 0..num_leaves {
            assert_eq!(
                &nodes[leaves_offset + i * 32..leaves_offset + (i + 1) * 32],
                &cpu[i][..],
                "fri leaf mismatch at i={i} (log_lde={log_lde})"
            );
        }
    }
}
