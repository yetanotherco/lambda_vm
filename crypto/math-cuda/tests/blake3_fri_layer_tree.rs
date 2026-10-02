//! Parity: the device BLAKE3 FRI-layer tree must equal the CPU tree node for
//! node — leaves and inner nodes alike.
//!
//! Mirror of `fri_layer_tree.rs` with the backend swapped. The CPU side is the
//! production `MerkleTree::build` over `PairBlake3Backend`, so what is compared
//! is the kernel pair against the real commitment path, not against a tree
//! builder written for the test.
//!
//! `blake3_leaves.rs` already pins the leaf layer alone; this adds the inner
//! nodes, which is where the level/tail launch split lives. Deep trees cross the
//! threshold where `blake3_merkle_level` hands over to `blake3_merkle_tail`, so
//! both kernels run.
//!
//! Needs a GPU.

mod blake3_reference;

use blake3_reference::expected_device_rounds;
use crypto::hash::blake3::BLAKE3_ROUNDS;
use crypto::merkle_tree::backends::types::PairBlake3Backend;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math_cuda::blake3::build_fri_layer_tree_from_evals_ext3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;
type Ext3 = Degree3GoldilocksExtensionField;

fn rand_ext3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
    ])
}

fn run_parity(log_num_leaves: u32, seed: u64) {
    assert_eq!(
        BLAKE3_ROUNDS,
        expected_device_rounds(),
        "crypto's blake3-6round and math-cuda's are out of lockstep"
    );

    let num_leaves = 1usize << log_num_leaves;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evals: Vec<Fp3> = (0..num_leaves * 2).map(|_| rand_ext3(&mut rng)).collect();

    let mut evals_u64 = Vec::with_capacity(evals.len() * 3);
    for e in &evals {
        evals_u64.push(*e.value()[0].value());
        evals_u64.push(*e.value()[1].value());
        evals_u64.push(*e.value()[2].value());
    }

    let leaves: Vec<[Fp3; 2]> = evals.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let cpu_tree = MerkleTree::<PairBlake3Backend<Ext3>>::build(&leaves).unwrap();
    let cpu_nodes = cpu_tree.nodes();

    let gpu_bytes = build_fri_layer_tree_from_evals_ext3(&evals_u64).unwrap();

    assert_eq!(cpu_nodes.len() * 32, gpu_bytes.len(), "node count");
    for (i, expected) in cpu_nodes.iter().enumerate() {
        assert_eq!(
            &gpu_bytes[i * 32..(i + 1) * 32],
            &expected[..],
            "node {i} mismatch at log_num_leaves={log_num_leaves}"
        );
    }
}

/// Small trees: every level fits the block width, so the tail kernel builds the
/// whole tree in one launch.
#[test]
fn blake3_fri_layer_tree_small() {
    for log in 1u32..=6 {
        run_parity(log, 100 + log as u64);
    }
}

/// Deep enough that the per-level kernel runs first and hands over to the tail
/// partway up — the launch path a real commit takes.
#[test]
fn blake3_fri_layer_tree_medium() {
    for log in [10u32, 12, 14] {
        run_parity(log, 500 + log as u64);
    }
}

#[test]
fn blake3_fri_layer_tree_large() {
    run_parity(18, 9999);
}
