//! Parity: GPU Merkle inner-tree construction must match the CPU
//! `crypto/crypto/src/merkle_tree/merkle.rs` `build_from_hashed_leaves`
//! (Keccak-256 pair hash at each level). Uses the prover's
//! `FieldElementVectorBackend<_, Keccak256, 32>` directly so any change to
//! the CPU tree builder is automatically exercised here.

use crypto::merkle_tree::backends::field_element_vector::FieldElementVectorBackend;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha3::Keccak256;

type CpuTree = MerkleTree<FieldElementVectorBackend<GoldilocksField, Keccak256, 32>>;

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

    // CPU reference: the prover's MerkleTree builder over the same backend.
    let cpu_tree = CpuTree::build_from_hashed_leaves(leaves).unwrap();
    let cpu_nodes = cpu_tree.nodes();

    for (i, c) in cpu_nodes.iter().enumerate() {
        let g = &gpu_nodes_bytes[i * 32..(i + 1) * 32];
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
fn merkle_tree_large() {
    run_parity(18, 9999);
}
