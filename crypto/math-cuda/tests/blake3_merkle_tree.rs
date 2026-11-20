//! Parity: the device BLAKE3 Merkle tree must equal the CPU tree node for node.
//!
//! The CPU side is the *production* tree walk — `MerkleTree::build_from_hashed_leaves`
//! over a backend whose only new code is `hash_new_parent` — so what this compares
//! is the parent compression and the node layout, not a second tree builder.
//! Both the per-level kernel and the single-block tail kernel are exercised: the
//! tail takes over once a level is no wider than the block, so a tree deep enough
//! to cross that threshold runs both, and the small trees run the tail alone.
//!
//! Mirrors `merkle_root_parity.rs` / `merkle_tree.rs` in structure. Needs a GPU.

mod blake3_reference;

use blake3_reference::{expected_device_rounds, merkle_parent};
use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math_cuda::blake3::build_merkle_tree_on_device;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The host parent hash under test, wrapped as a Merkle backend so the CPU
/// reference is the production tree walk rather than a hand-rolled one.
///
/// `hash_data` is unreachable here — leaves are supplied already hashed — and is
/// wired to the same parent function rather than to `unimplemented!()` so the
/// backend stays a total function if a later test does call it.
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

fn random_leaves(count: usize, seed: u64) -> Vec<[u8; 32]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| core::array::from_fn(|_| rng.r#gen::<u8>()))
        .collect()
}

fn run_parity(log_num_leaves: u32, seed: u64) {
    let num_leaves = 1usize << log_num_leaves;
    let leaves = random_leaves(num_leaves, seed);

    let cpu = MerkleTree::<Blake3ParentBackend>::build_from_hashed_leaves(leaves.clone()).unwrap();
    let cpu_nodes = cpu.nodes();

    let flat: Vec<u8> = leaves.iter().flatten().copied().collect();
    let gpu = build_merkle_tree_on_device(&flat).unwrap();

    assert_eq!(cpu_nodes.len() * 32, gpu.len(), "node count");
    for (i, expected) in cpu_nodes.iter().enumerate() {
        assert_eq!(
            &gpu[i * 32..(i + 1) * 32],
            &expected[..],
            "node {i} mismatch at log_num_leaves = {log_num_leaves}"
        );
    }
}

/// Small trees: every level fits the block width, so the tail kernel builds the
/// whole tree in one launch.
#[test]
fn blake3_merkle_tree_small() {
    for log in 1u32..=8 {
        run_parity(log, 300 + log as u64);
    }
}

/// Deep enough that the per-level kernel runs first and hands over to the tail
/// partway up — the launch path a real commit takes.
#[test]
fn blake3_merkle_tree_medium() {
    for log in [10u32, 12, 14] {
        run_parity(log, 700 + log as u64);
    }
}

#[test]
fn blake3_merkle_tree_large() {
    run_parity(18, 4242);
}

/// The parent is `hash_bytes(left ‖ right)` — a plain library call at 7 rounds,
/// which is the property that makes the framing (`h = IV`, `t = 0`,
/// `block_len = 64`, `flags = CHUNK_START|CHUNK_END|ROOT`) externally anchored
/// rather than merely self-consistent.
///
/// Only meaningful when the kernels are built for 7 rounds; under
/// `blake3-6round` there is nothing in the world that recomputes the parent, which
/// is exactly PA-PLAN §1.6's premise.
#[test]
fn the_parent_is_the_blake3_crate_at_seven_rounds() {
    if expected_device_rounds() != 7 {
        return;
    }
    let leaves = random_leaves(2, 31337);
    let flat: Vec<u8> = leaves.iter().flatten().copied().collect();
    let gpu = build_merkle_tree_on_device(&flat).unwrap();

    let mut msg = Vec::with_capacity(64);
    msg.extend_from_slice(&leaves[0]);
    msg.extend_from_slice(&leaves[1]);
    assert_eq!(
        &gpu[0..32],
        blake3::hash(&msg).as_bytes(),
        "a two-leaf root must be blake3::hash(left ‖ right)"
    );
}
