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
