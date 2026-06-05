//! Parity: GPU fused `evaluate_poly_coset_batch_ext3_into_with_merkle_tree`
//! (LDE + row-pair Keccak leaves + Merkle inner tree) against the same CPU
//! pipeline produced by `commit_composition_polynomial`.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math::polynomial::Polynomial;
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

fn offset_weights(n: usize, offset: u64) -> Vec<u64> {
    let mut w = Vec::with_capacity(n);
    let mut cur = 1u64;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &offset);
    }
    w
}

fn rand_ext3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
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

fn u64s_to_ext3(raw: &[u64]) -> Vec<Fp3> {
    let mut out = Vec::with_capacity(raw.len() / 3);
    for i in 0..raw.len() / 3 {
        out.push(Fp3::new([
            Fp::from_raw(raw[i * 3]),
            Fp::from_raw(raw[i * 3 + 1]),
            Fp::from_raw(raw[i * 3 + 2]),
        ]));
    }
    out
}

fn canon_ext3(e: &Fp3) -> [u64; 3] {
    [
        GoldilocksField::canonical(e.value()[0].value()),
        GoldilocksField::canonical(e.value()[1].value()),
        GoldilocksField::canonical(e.value()[2].value()),
    ]
}

/// CPU: evaluate polynomial on coset via `Polynomial::evaluate_offset_fft`.
fn cpu_evaluate(coefs: &[Fp3], blowup: usize, offset: &Fp) -> Vec<Fp3> {
    let poly = Polynomial::new(coefs);
    Polynomial::evaluate_offset_fft::<GoldilocksField>(&poly, blowup, None, offset).unwrap()
}

fn cpu_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(left);
    h.update(right);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

/// CPU: `commit_composition_polynomial`-style tree root over num_rows/2 leaves.
fn cpu_tree_nodes(parts: &[Vec<Fp3>]) -> Vec<[u8; 32]> {
    let num_rows = parts[0].len();
    let num_parts = parts.len();
    let num_leaves = num_rows / 2;
    assert!(num_leaves.is_power_of_two() && num_leaves >= 1);
    let byte_len = 24;

    let hashed_leaves: Vec<[u8; 32]> = (0..num_leaves)
        .map(|leaf_idx| {
            let br_0 = reverse_index(2 * leaf_idx as u64, num_rows as u64) as usize;
            let br_1 = reverse_index(2 * leaf_idx as u64 + 1, num_rows as u64) as usize;
            let total_bytes = 2 * num_parts * byte_len;
            let mut buf = vec![0u8; total_bytes];
            let mut offset = 0;
            for part in parts.iter() {
                part[br_0].write_bytes_be(&mut buf[offset..offset + byte_len]);
                offset += byte_len;
            }
            for part in parts.iter() {
                part[br_1].write_bytes_be(&mut buf[offset..offset + byte_len]);
                offset += byte_len;
            }
            let mut h = Keccak256::new();
            h.update(&buf);
            let mut r = [0u8; 32];
            r.copy_from_slice(&h.finalize());
            r
        })
        .collect();

    let total = 2 * num_leaves - 1;
    let mut nodes: Vec<[u8; 32]> = vec![[0u8; 32]; total];
    for (i, leaf) in hashed_leaves.iter().enumerate() {
        nodes[num_leaves - 1 + i] = *leaf;
    }
    let mut level_begin = num_leaves - 1;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        for j in 0..n_pairs {
            let left = nodes[level_begin + 2 * j];
            let right = nodes[level_begin + 2 * j + 1];
            nodes[new_begin + j] = cpu_hash_pair(&left, &right);
        }
        level_begin = new_begin;
    }
    nodes
}

fn run_parity(log_n: u32, blowup: usize, num_parts: usize, seed: u64) {
    let n = 1usize << log_n;
    let lde_size = n * blowup;
    assert!(lde_size >= 2);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Random ext3 coefficient vectors per part.
    let parts_cpu: Vec<Vec<Fp3>> = (0..num_parts)
        .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
        .collect();

    // CPU LDE via evaluate_offset_fft, then CPU tree.
    let offset_u64 = rng.r#gen::<u64>() | 1;
    let offset = Fp::from_raw(offset_u64);
    let cpu_lde_parts: Vec<Vec<Fp3>> = parts_cpu
        .iter()
        .map(|c| cpu_evaluate(c, blowup, &offset))
        .collect();
    let cpu_nodes = cpu_tree_nodes(&cpu_lde_parts);

    // GPU fused call.
    let weights = offset_weights(n, offset_u64);
    let coefs_u64: Vec<Vec<u64>> = parts_cpu.iter().map(|c| ext3_to_u64s(c)).collect();
    let coefs_slices: Vec<&[u64]> = coefs_u64.iter().map(|v| v.as_slice()).collect();
    let mut outputs_raw: Vec<Vec<u64>> = (0..num_parts).map(|_| vec![0u64; 3 * lde_size]).collect();
    let mut outputs_slices: Vec<&mut [u64]> =
        outputs_raw.iter_mut().map(|v| v.as_mut_slice()).collect();
    // R2 leaves are row-pairs: num_leaves = lde_size / 2, so
    // tight_total_nodes = 2 * num_leaves - 1 = lde_size - 1.
    let total_nodes = lde_size - 1;
    let mut nodes_bytes = vec![0u8; total_nodes * 32];

    math_cuda::lde::evaluate_poly_coset_batch_ext3_into_with_merkle_tree(
        &coefs_slices,
        n,
        blowup,
        &weights,
        &mut outputs_slices,
        &mut nodes_bytes,
    )
    .unwrap();

    // Compare LDE parts.
    for (c, cpu_col) in cpu_lde_parts.iter().enumerate() {
        let gpu_col = u64s_to_ext3(&outputs_raw[c]);
        for i in 0..lde_size {
            assert_eq!(
                canon_ext3(&gpu_col[i]),
                canon_ext3(&cpu_col[i]),
                "LDE mismatch part {c} row {i} log_n={log_n} blowup={blowup}"
            );
        }
    }

    // Compare tree nodes. GPU writes `2*num_leaves - 1 = lde_size - 1` nodes.
    let num_leaves = lde_size / 2;
    let tight_total = 2 * num_leaves - 1;
    assert_eq!(cpu_nodes.len(), tight_total);
    for i in 0..tight_total {
        let g = &nodes_bytes[i * 32..(i + 1) * 32];
        let c = &cpu_nodes[i];
        assert_eq!(
            g, c,
            "tree node {i} mismatch at log_n={log_n} blowup={blowup} parts={num_parts}"
        );
    }
}

#[test]
fn comp_poly_tree_small() {
    for log_n in 2u32..=5 {
        for &blowup in &[2usize, 4, 8] {
            for &parts in &[1usize, 2, 4] {
                run_parity(
                    log_n,
                    blowup,
                    parts,
                    1000 + log_n as u64 * 31 + parts as u64,
                );
            }
        }
    }
}

#[test]
fn comp_poly_tree_medium() {
    for &(log_n, blowup, parts) in &[(10u32, 4usize, 4usize), (12, 2, 3)] {
        run_parity(
            log_n,
            blowup,
            parts,
            2000 + log_n as u64 * 11 + parts as u64,
        );
    }
}

#[test]
fn comp_poly_tree_large() {
    run_parity(14, 2, 4, 9999);
}
