//! Parity: GPU Merkle inner-tree construction must match the CPU
//! `crypto/crypto/src/merkle_tree/merkle.rs` `build_from_hashed_leaves`
//! (Keccak-256 pair hash at each level).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha3::{Digest, Keccak256};

fn cpu_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(left);
    h.update(right);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

/// CPU reference: same algorithm as `build_from_hashed_leaves`.
fn cpu_merkle_nodes(leaves: &[[u8; 32]]) -> Vec<[u8; 32]> {
    let leaves_len = leaves.len();
    assert!(leaves_len.is_power_of_two() && leaves_len >= 2);
    let total = 2 * leaves_len - 1;

    let mut nodes: Vec<[u8; 32]> = vec![[0u8; 32]; total];
    for (i, leaf) in leaves.iter().enumerate() {
        nodes[leaves_len - 1 + i] = *leaf;
    }

    let mut level_begin = leaves_len - 1;
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

fn run_parity(log_n: u32, seed: u64) {
    let leaves_len = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let leaves: Vec<[u8; 32]> = (0..leaves_len)
        .map(|_| {
            let mut arr = [0u8; 32];
            rng.fill(&mut arr[..]);
            arr
        })
        .collect();

    // Flat byte layout for the GPU entry point.
    let mut flat = Vec::with_capacity(leaves_len * 32);
    for l in &leaves {
        flat.extend_from_slice(l);
    }

    let gpu_nodes_bytes = math_cuda::merkle::build_merkle_tree_on_device(&flat).unwrap();
    assert_eq!(gpu_nodes_bytes.len(), (2 * leaves_len - 1) * 32);

    let cpu_nodes = cpu_merkle_nodes(&leaves);

    for i in 0..cpu_nodes.len() {
        let g = &gpu_nodes_bytes[i * 32..(i + 1) * 32];
        let c = &cpu_nodes[i];
        assert_eq!(
            g, c,
            "node {i} mismatch at log_n={log_n} (cpu={c:?}, gpu={g:?})"
        );
    }
}

#[test]
fn merkle_tree_small() {
    for log_n in 1u32..=6 {
        run_parity(log_n, 100 + log_n as u64);
    }
}

#[test]
fn merkle_tree_medium() {
    for log_n in [10u32, 12, 14] {
        run_parity(log_n, 500 + log_n as u64);
    }
}

#[test]
fn merkle_tree_large() {
    run_parity(18, 9999);
}
