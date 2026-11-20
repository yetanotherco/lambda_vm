//! Parity: authentication paths gathered from a BLAKE3 tree must be the paths
//! the CPU `MerkleTree::get_proof_by_pos` returns.
//!
//! `merkle_gather_paths` is HASH-AGNOSTIC — it copies sibling nodes and never
//! hashes — so PA-PLAN §6.1 correctly says it needs no BLAKE3 twin, and none is
//! written. What is not free is the claim that it walks a BLAKE3 tree correctly:
//! that depends on `blake3::build_merkle_tree_on_device` laying nodes out in the
//! same order the keccak builder does, which is a property of the new code. This
//! file is that check, and it is why the gather is reused rather than twinned
//! *and tested*, rather than reused on the strength of the argument alone.
//!
//! Mirror of `merkle_gather.rs` with the tree builder swapped.
//!
//! Needs a GPU.

mod blake3_reference;

use blake3_reference::{expected_device_rounds, merkle_parent};
use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The host parent hash as a Merkle backend, so the CPU reference is the
/// production tree walk. Leaves arrive already hashed, so `hash_data` is
/// unreachable; it is wired to the same parent function rather than to
/// `unimplemented!()` so the backend stays a total function.
#[derive(Clone, Default)]
struct Blake3ParentBackend;

impl IsMerkleTreeBackend for Blake3ParentBackend {
    type Node = [u8; 32];
    type Data = [u8; 32];

    fn hash_data(leaf: &Self::Data) -> Self::Node {
        merkle_parent(leaf, leaf, expected_device_rounds())
    }

    fn hash_new_parent(a: &Self::Node, b: &Self::Node) -> Self::Node {
        merkle_parent(a, b, expected_device_rounds())
    }
}

fn run_gather_parity(log_n: u32, seed: u64) {
    let leaves_len = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let leaves: Vec<[u8; 32]> = (0..leaves_len)
        .map(|_| core::array::from_fn(|_| rng.r#gen::<u8>()))
        .collect();
    let flat: Vec<u8> = leaves.iter().flatten().copied().collect();

    // Build the BLAKE3 tree on device, then upload its nodes back as the
    // resident buffer the gather reads.
    let gpu_nodes_bytes = math_cuda::blake3::build_merkle_tree_on_device(&flat).unwrap();
    let cpu_tree = MerkleTree::<Blake3ParentBackend>::build_from_hashed_leaves(leaves).unwrap();

    // A spread of positions: first, last, and random interior ones.
    let mut positions: Vec<u32> = vec![0, (leaves_len - 1) as u32];
    let mut r = ChaCha8Rng::seed_from_u64(seed ^ 0xabcd);
    for _ in 0..16usize.min(leaves_len) {
        positions.push(r.gen_range(0..leaves_len) as u32);
    }

    let be = math_cuda::device::backend().unwrap();
    let stream = be.next_stream();
    let nodes_dev = stream.clone_htod(&gpu_nodes_bytes).unwrap();
    stream.synchronize().unwrap();

    let depth = log_n as usize;
    let paths =
        math_cuda::merkle::gather_merkle_paths_dev(&nodes_dev, leaves_len, &positions, &stream)
            .unwrap();
    assert_eq!(paths.len(), positions.len() * depth * 32);

    for (q, &pos) in positions.iter().enumerate() {
        let cpu_proof = cpu_tree.get_proof_by_pos(pos as usize).unwrap();
        assert_eq!(
            cpu_proof.merkle_path.len(),
            depth,
            "depth mismatch at log_n={log_n} pos={pos}"
        );
        for (level, cpu_node) in cpu_proof.merkle_path.iter().enumerate() {
            assert_eq!(
                &paths[(q * depth + level) * 32..(q * depth + level + 1) * 32],
                &cpu_node[..],
                "path node mismatch: log_n={log_n} pos={pos} level={level}"
            );
        }
    }
}

#[test]
fn blake3_merkle_gather_small() {
    for log_n in 1u32..=6 {
        run_gather_parity(log_n, 200 + log_n as u64);
    }
}

#[test]
fn blake3_merkle_gather_large() {
    run_gather_parity(18, 7777);
}
