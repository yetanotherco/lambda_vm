//! Parity: the device BLAKE3 leaf kernels must reproduce the CPU prover's leaf
//! hashes byte for byte.
//!
//! Structural mirror of `keccak_leaves.rs`, and deliberately so: the leaf BYTE
//! layout does not move under P-a. `leaves_bit_reversed_grouped`
//! (`crypto/stark/src/commitment.rs:55`) serializes each element in canonical
//! big-endian, concatenates, and hashes the buffer once — the same bytes for
//! both hashes. What changes is only the hash over them, so the CPU reference
//! here is the *production* leaf function instantiated at the BLAKE3 backend
//! rather than a second implementation written for the test.
//!
//! That makes each assertion below a check of two things at once: that the
//! kernel's read pattern (bit reversal, column order, component order, row-pair
//! ordering) matches the CPU's, and that the device `Blake3Chain` matches the
//! host one over multi-block messages.
//!
//! Needs a GPU. See `RESUME-TRACKG.md` for the run command.

mod blake3_reference;

use blake3_reference::expected_device_rounds;
use crypto::hash::blake3::BLAKE3_ROUNDS;
use crypto::merkle_tree::backends::types::{BatchBlake3Backend, PairBlake3Backend};
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::traits::{AsBytes, ByteConversion};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::commitment::leaves_bit_reversed_grouped;
use stark::config::Commitment;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

/// The CPU leaf hashes for `columns`, through the production leaf function at
/// the BLAKE3 batched backend.
fn cpu_leaves<E>(columns: &[Vec<FieldElement<E>>], rows_per_leaf: usize) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    leaves_bit_reversed_grouped::<E, BatchBlake3Backend<E>>(columns, rows_per_leaf)
}

/// ★ LOCKSTEP GUARD. `crypto`'s `blake3-6round` and `math-cuda`'s are separate
/// features and nothing forces them equal. Out of lockstep, every assertion in
/// this file compares a 6-round device tree against a 7-round host one (or the
/// reverse) and fails with a wall of unequal bytes that says nothing about the
/// cause. Failing here first names it.
///
/// This is the same guard `blake3_reference`'s parent test carries, repeated
/// because a leaf-kernel failure has the same ambiguity and a developer running
/// only this file would not see the other one.
fn assert_round_lockstep() {
    assert_eq!(
        BLAKE3_ROUNDS,
        expected_device_rounds(),
        "crypto's blake3-6round and math-cuda's are out of lockstep: the GPU \
         kernels would commit under a different hash than the CPU backend. Set \
         both features or neither."
    );
}

fn rand_base(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}

fn rand_ext3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
    ])
}

/// Base columns into the contiguous `[col * stride + row]` slab the kernels read
/// — the layout `coset_lde_batch_base_into` writes to pinned staging.
fn base_slabs(columns: &[Vec<Fp>], n: usize) -> Vec<u64> {
    let mut flat = vec![0u64; columns.len() * n];
    for (c, col) in columns.iter().enumerate() {
        for (r, e) in col.iter().enumerate() {
            flat[c * n + r] = *e.value();
        }
    }
    flat
}

/// Ext3 columns into three base slabs per column: `[col*3 + k]`, each a
/// contiguous slab of `n` u64s.
fn ext3_slabs(columns: &[Vec<Fp3>], n: usize) -> Vec<u64> {
    let mut flat = vec![0u64; columns.len() * 3 * n];
    for (c, col) in columns.iter().enumerate() {
        for (r, e) in col.iter().enumerate() {
            flat[(c * 3) * n + r] = *e.value()[0].value();
            flat[(c * 3 + 1) * n + r] = *e.value()[1].value();
            flat[(c * 3 + 2) * n + r] = *e.value()[2].value();
        }
    }
    flat
}

fn assert_leaves_eq(gpu: &[u8], cpu: &[Commitment], what: &str) {
    assert_eq!(gpu.len(), cpu.len() * 32, "{what}: leaf count");
    for (i, expected) in cpu.iter().enumerate() {
        assert_eq!(
            &gpu[i * 32..(i + 1) * 32],
            &expected[..],
            "{what}: leaf {i} mismatch"
        );
    }
}

/// Column counts are chosen to straddle the 64-byte block boundary in both
/// directions: 8 base elements fill a block exactly, so 1/5/17/41 columns give
/// leaves that end mid-block, on a boundary, and several blocks in. That is
/// where a chaining bug lives — a kernel that compressed eagerly on fill, or
/// mis-set `CHUNK_START` on a later block, agrees with the host at one column
/// count and not at the next.
#[test]
fn blake3_leaves_base_matches_cpu() {
    assert_round_lockstep();
    for log_n in [4u32, 6, 8, 10, 12] {
        for num_cols in [1usize, 5, 8, 17, 41] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(100 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp>> = (0..num_cols)
                .map(|_| (0..n).map(|_| rand_base(&mut rng)).collect())
                .collect();

            let cpu = cpu_leaves(&columns, 1);
            let flat = base_slabs(&columns, n);
            let gpu = math_cuda::blake3::leaves_base(&flat, n, num_cols, n, 1).unwrap();
            assert_leaves_eq(&gpu, &cpu, &format!("base log_n={log_n} cols={num_cols}"));
        }
    }
}

#[test]
fn blake3_leaves_base_row_pair_matches_cpu() {
    assert_round_lockstep();
    for log_n in [4u32, 6, 8, 10, 12] {
        for num_cols in [1usize, 5, 8, 17, 41] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(500 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp>> = (0..num_cols)
                .map(|_| (0..n).map(|_| rand_base(&mut rng)).collect())
                .collect();

            let cpu = cpu_leaves(&columns, 2);
            assert_eq!(cpu.len(), n / 2);
            let flat = base_slabs(&columns, n);
            let gpu = math_cuda::blake3::leaves_base(&flat, n, num_cols, n, 2).unwrap();
            assert_leaves_eq(
                &gpu,
                &cpu,
                &format!("base row-pair log_n={log_n} cols={num_cols}"),
            );
        }
    }
}

/// Ext3 elements are three felts = six words, so they straddle block boundaries
/// on most column counts rather than only on a few — the case the word-granular
/// (rather than element-granular) block builder exists for.
#[test]
fn blake3_leaves_ext3_matches_cpu() {
    assert_round_lockstep();
    for log_n in [4u32, 6, 8, 10] {
        for num_cols in [1usize, 3, 11, 20] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(200 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp3>> = (0..num_cols)
                .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
                .collect();

            let cpu = cpu_leaves(&columns, 1);
            let flat = ext3_slabs(&columns, n);
            let gpu = math_cuda::blake3::leaves_ext3(&flat, n, num_cols, n, 1).unwrap();
            assert_leaves_eq(&gpu, &cpu, &format!("ext3 log_n={log_n} cols={num_cols}"));
        }
    }
}

#[test]
fn blake3_leaves_ext3_row_pair_matches_cpu() {
    assert_round_lockstep();
    for log_n in [4u32, 6, 8, 10] {
        for num_cols in [1usize, 3, 11, 20] {
            let n = 1 << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(600 + log_n as u64 + num_cols as u64);
            let columns: Vec<Vec<Fp3>> = (0..num_cols)
                .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
                .collect();

            let cpu = cpu_leaves(&columns, 2);
            assert_eq!(cpu.len(), n / 2);
            let flat = ext3_slabs(&columns, n);
            let gpu = math_cuda::blake3::leaves_ext3(&flat, n, num_cols, n, 2).unwrap();
            assert_leaves_eq(
                &gpu,
                &cpu,
                &format!("ext3 row-pair log_n={log_n} cols={num_cols}"),
            );
        }
    }
}

/// FRI leaves are 48 bytes — under one block — so this is the chain's degenerate
/// single-compression case: `flags = 0x0B`, `block_len = 48`. Same shape as a
/// Merkle parent at a different length, which is why it is worth pinning
/// separately from the multi-block leaves above.
#[test]
fn blake3_fri_leaves_matches_cpu() {
    assert_round_lockstep();
    for log_lde in [2u32, 4, 6, 8, 10, 12] {
        let lde_size = 1usize << log_lde;
        let mut rng = ChaCha8Rng::seed_from_u64(400 + log_lde as u64);
        let evals: Vec<Fp3> = (0..lde_size).map(|_| rand_ext3(&mut rng)).collect();

        let cpu: Vec<[u8; 32]> = evals
            .chunks_exact(2)
            .map(|c| PairBlake3Backend::<Degree3GoldilocksExtensionField>::hash_data(&[c[0], c[1]]))
            .collect();

        let mut evals_interleaved = vec![0u64; 3 * lde_size];
        for (i, e) in evals.iter().enumerate() {
            evals_interleaved[i * 3] = *e.value()[0].value();
            evals_interleaved[i * 3 + 1] = *e.value()[1].value();
            evals_interleaved[i * 3 + 2] = *e.value()[2].value();
        }
        let nodes =
            math_cuda::blake3::build_fri_layer_tree_from_evals_ext3(&evals_interleaved).unwrap();
        let num_leaves = lde_size / 2;
        let leaves_offset = (num_leaves - 1) * 32;
        assert_leaves_eq(
            &nodes[leaves_offset..leaves_offset + num_leaves * 32],
            &cpu,
            &format!("fri log_lde={log_lde}"),
        );
    }
}

/// The comp-poly kernel through the production keep path, checked at the leaf
/// layer: the resident node buffer's leaf half must be the CPU's row-pair leaves.
#[test]
fn blake3_comp_poly_leaves_matches_cpu() {
    assert_round_lockstep();
    for log_lde in [2u32, 4, 6, 8, 10, 12] {
        for num_parts in [1usize, 2, 5, 17] {
            let lde_size = 1usize << log_lde;
            let mut rng = ChaCha8Rng::seed_from_u64(300 + log_lde as u64 + num_parts as u64);
            let parts: Vec<Vec<Fp3>> = (0..num_parts)
                .map(|_| (0..lde_size).map(|_| rand_ext3(&mut rng)).collect())
                .collect();
            let cpu = cpu_leaves(&parts, 2);

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

            let tree = math_cuda::blake3::build_comp_poly_tree_from_evals_ext3_keep(&parts_slices)
                .unwrap();
            let be = math_cuda::device::backend().unwrap();
            let stream = be.next_stream();
            let nodes: Vec<u8> = stream.clone_dtoh(&*tree.nodes).unwrap();
            let num_leaves = lde_size / 2;
            let leaves_offset = (num_leaves - 1) * 32;
            assert_leaves_eq(
                &nodes[leaves_offset..leaves_offset + num_leaves * 32],
                &cpu,
                &format!("comp-poly log_lde={log_lde} parts={num_parts}"),
            );
        }
    }
}

/// Row-major row-pair leaves. The CPU reference is the same
/// `leaves_bit_reversed_grouped(.., 2)` — over the column-major view of the same
/// buffer, which is exactly the equivalence the row-major kernel exists to
/// exploit (`commit_rows_bit_reversed` reads rows contiguously instead of
/// transposing).
#[test]
fn blake3_leaves_row_major_row_pair_matches_cpu() {
    assert_round_lockstep();
    for log_n in [4u32, 6, 8, 10] {
        for m in [1usize, 5, 8, 17] {
            let n = 1usize << log_n;
            let mut rng = ChaCha8Rng::seed_from_u64(800 + log_n as u64 + m as u64);
            // Row-major: row r occupies `data[r*m .. r*m + m]`.
            let data: Vec<Fp> = (0..n * m).map(|_| rand_base(&mut rng)).collect();

            let columns: Vec<Vec<Fp>> = (0..m)
                .map(|c| (0..n).map(|r| data[r * m + c]).collect())
                .collect();
            let cpu = cpu_leaves(&columns, 2);

            let raw: Vec<u64> = data.iter().map(|e| *e.value()).collect();
            let gpu = math_cuda::blake3::leaves_base_row_major_row_pair(&raw, m, n).unwrap();
            assert_leaves_eq(&gpu, &cpu, &format!("row-major log_n={log_n} m={m}"));
        }
    }
}

/// The column-range variant, which is how preprocessed tables commit their
/// precomputed and multiplicity ranges to separate trees over one LDE. The
/// reference is the same function over just those columns — so this pins that
/// `m` stays the full row stride while only `[col_start, col_end)` is hashed.
#[test]
fn blake3_leaves_row_major_row_pair_range_matches_cpu() {
    assert_round_lockstep();
    for log_n in [4u32, 6, 8, 10] {
        let n = 1usize << log_n;
        let m = 13usize;
        let mut rng = ChaCha8Rng::seed_from_u64(900 + log_n as u64);
        let data: Vec<Fp> = (0..n * m).map(|_| rand_base(&mut rng)).collect();

        // A split that is not on a block boundary either side of it.
        for (col_start, col_end) in [(0usize, 5usize), (5, 13), (0, 13), (3, 4)] {
            let columns: Vec<Vec<Fp>> = (col_start..col_end)
                .map(|c| (0..n).map(|r| data[r * m + c]).collect())
                .collect();
            let cpu = cpu_leaves(&columns, 2);

            let raw: Vec<u64> = data.iter().map(|e| *e.value()).collect();
            let gpu = math_cuda::blake3::leaves_base_row_major_row_pair_range(
                &raw, m, col_start, col_end, n,
            )
            .unwrap();
            assert_leaves_eq(
                &gpu,
                &cpu,
                &format!("row-major range log_n={log_n} cols=[{col_start},{col_end})"),
            );
        }
    }
}

/// ★ NEGATIVE CONTROL for the whole file.
///
/// Every test above asserts device == host. All of them would pass just as well
/// if both sides were a constant, or if the kernel ignored its input entirely
/// and the CPU reference happened to be compared against itself. This asserts
/// the leaves actually depend on the data: two column sets differing in one
/// element must give different leaves, and distinct rows must give distinct
/// leaves.
#[test]
fn leaves_depend_on_the_data() {
    assert_round_lockstep();
    let n = 64usize;
    let num_cols = 5usize;
    let mut rng = ChaCha8Rng::seed_from_u64(4242);
    let columns: Vec<Vec<Fp>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rand_base(&mut rng)).collect())
        .collect();

    let flat = base_slabs(&columns, n);
    let a = math_cuda::blake3::leaves_base(&flat, n, num_cols, n, 1).unwrap();

    // Perturb one element and re-hash.
    let mut perturbed = columns.clone();
    perturbed[2][7] += Fp::from(1u64);
    let flat2 = base_slabs(&perturbed, n);
    let b = math_cuda::blake3::leaves_base(&flat2, n, num_cols, n, 1).unwrap();
    assert_ne!(a, b, "a one-element change must move some leaf");

    // And the leaves are not all the same digest.
    let first = &a[0..32];
    assert!(
        a.chunks_exact(32).any(|c| c != first),
        "all leaves identical — the kernel is not reading its row index"
    );
}
