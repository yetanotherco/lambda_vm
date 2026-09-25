//! Parity: GPU `build_fri_layer_tree_from_evals_ext3` vs CPU
//! `FriLayerMerkleTree::build` (PairKeccak256 backend over ext3 pairs).

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math_cuda::merkle::build_fri_layer_tree_from_evals_ext3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::config::KeccakFriLayerMerkleTree as FriLayerMerkleTree;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

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

fn run_parity(log_num_leaves: u32, seed: u64) {
    let num_leaves = 1usize << log_num_leaves;
    let num_evals = num_leaves * 2;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evals: Vec<Fp3> = (0..num_evals).map(|_| rand_ext3(&mut rng)).collect();
    let evals_u64 = ext3_to_u64s(&evals);

    // CPU reference: the production MerkleTree builder over PairKeccak256.
    let leaves: Vec<[Fp3; 2]> = evals.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let cpu_tree = FriLayerMerkleTree::<Degree3GoldilocksExtensionField>::build(&leaves).unwrap();
    let cpu_nodes = cpu_tree.nodes();

    let gpu_bytes = build_fri_layer_tree_from_evals_ext3(&evals_u64).unwrap();

    assert_eq!(cpu_nodes.len() * 32, gpu_bytes.len());
    for i in 0..cpu_nodes.len() {
        let g = &gpu_bytes[i * 32..(i + 1) * 32];
        let c = &cpu_nodes[i];
        assert_eq!(g, c, "node {i} mismatch at log_num_leaves={log_num_leaves}");
    }
}

#[test]
fn fri_layer_tree_small() {
    for log in 1u32..=6 {
        run_parity(log, 100 + log as u64);
    }
}

#[test]
fn fri_layer_tree_medium() {
    for log in [10u32, 12, 14] {
        run_parity(log, 500 + log as u64);
    }
}

#[test]
fn fri_layer_tree_large() {
    run_parity(18, 9999);
}
