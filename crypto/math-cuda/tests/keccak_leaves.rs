//! Parity: GPU Keccak-256 leaf hashes must match CPU
//! `FieldElementVectorBackend::<F, Keccak256, 32>::hash_data` applied to
//! bit-reversed rows (same pattern as `commit_columns_bit_reversed` in the
//! stark prover).

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::traits::ByteConversion;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha3::{Digest, Keccak256};

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn reverse_index(i: u64, n: u64) -> u64 {
    let log_n = n.trailing_zeros();
    i.reverse_bits() >> (64 - log_n)
}

fn cpu_leaves_base(columns: &[Vec<Fp>]) -> Vec<[u8; 32]> {
    let num_rows = columns[0].len();
    let num_cols = columns.len();
    let byte_len = 8;
    (0..num_rows)
        .map(|row_idx| {
            let br = reverse_index(row_idx as u64, num_rows as u64) as usize;
            let mut buf = vec![0u8; num_cols * byte_len];
            for c in 0..num_cols {
                columns[c][br].write_bytes_be(&mut buf[c * byte_len..(c + 1) * byte_len]);
            }
            let mut h = Keccak256::new();
            h.update(&buf);
            let mut out = [0u8; 32];
            out.copy_from_slice(&h.finalize());
            out
        })
        .collect()
}

fn cpu_leaves_ext3(columns: &[Vec<Fp3>]) -> Vec<[u8; 32]> {
    let num_rows = columns[0].len();
    let num_cols = columns.len();
    let byte_len = 24;
    (0..num_rows)
        .map(|row_idx| {
            let br = reverse_index(row_idx as u64, num_rows as u64) as usize;
            let mut buf = vec![0u8; num_cols * byte_len];
            for c in 0..num_cols {
                columns[c][br].write_bytes_be(&mut buf[c * byte_len..(c + 1) * byte_len]);
            }
            let mut h = Keccak256::new();
            h.update(&buf);
            let mut out = [0u8; 32];
            out.copy_from_slice(&h.finalize());
            out
        })
        .collect()
}

#[test]
fn keccak_leaves_base_matches_cpu() {
    for log_n in [4u32, 6, 8, 10, 12] {
        for num_cols in [1usize, 5, 17, 41] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(100 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fp::from_raw(rng.r#gen::<u64>())).collect())
                .collect();

            let cpu = cpu_leaves_base(&columns);

            // Flatten columns into a contiguous base slab layout matching
            // `coset_lde_batch_base_into`'s pinned staging format:
            // `[col * stride + row]`. Use stride = num_rows for compactness.
            let mut flat = vec![0u64; num_cols * n];
            for (c, col) in columns.iter().enumerate() {
                for (r, e) in col.iter().enumerate() {
                    flat[c * n + r] = *e.value();
                }
            }
            let gpu = math_cuda::merkle::keccak_leaves_base(&flat, n, num_cols, n).unwrap();
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

// Row-pair leaves for the R2 composition-polynomial commit. For each leaf i:
//   br_0 = bit_reverse(2*i, log_lde),  br_1 = bit_reverse(2*i+1, log_lde)
// hash is Keccak256 of BE bytes of every part's ext3 value at br_0 then br_1
// (matching `commit_composition_polynomial` on the CPU side).
fn cpu_leaves_comp_poly(parts: &[Vec<Fp3>]) -> Vec<[u8; 32]> {
    let lde_size = parts[0].len();
    let num_parts = parts.len();
    let num_leaves = lde_size / 2;
    let byte_len = 24;
    (0..num_leaves)
        .map(|i| {
            let br_0 = reverse_index((2 * i) as u64, lde_size as u64) as usize;
            let br_1 = reverse_index((2 * i + 1) as u64, lde_size as u64) as usize;
            let mut buf = vec![0u8; 2 * num_parts * byte_len];
            for (p, part) in parts.iter().enumerate() {
                part[br_0].write_bytes_be(&mut buf[p * byte_len..(p + 1) * byte_len]);
            }
            let off = num_parts * byte_len;
            for (p, part) in parts.iter().enumerate() {
                part[br_1].write_bytes_be(&mut buf[off + p * byte_len..off + (p + 1) * byte_len]);
            }
            let mut h = Keccak256::new();
            h.update(&buf);
            let mut out = [0u8; 32];
            out.copy_from_slice(&h.finalize());
            out
        })
        .collect()
}

// FRI leaves: each leaf hashes 2 consecutive ext3 evals, no bit reversal.
fn cpu_leaves_fri(evals: &[Fp3]) -> Vec<[u8; 32]> {
    let num_leaves = evals.len() / 2;
    let byte_len = 24;
    (0..num_leaves)
        .map(|i| {
            let mut buf = vec![0u8; 2 * byte_len];
            evals[2 * i].write_bytes_be(&mut buf[..byte_len]);
            evals[2 * i + 1].write_bytes_be(&mut buf[byte_len..]);
            let mut h = Keccak256::new();
            h.update(&buf);
            let mut out = [0u8; 32];
            out.copy_from_slice(&h.finalize());
            out
        })
        .collect()
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
            let cpu = cpu_leaves_comp_poly(&parts);

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

            let nodes =
                math_cuda::merkle::build_comp_poly_tree_from_evals_ext3(&parts_slices).unwrap();
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
        let cpu = cpu_leaves_fri(&evals);

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

            let cpu = cpu_leaves_ext3(&columns);

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
            let gpu = math_cuda::merkle::keccak_leaves_ext3(&flat, n, num_cols, n).unwrap();
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
