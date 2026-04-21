//! Parity: GPU `build_fri_layer_tree_from_evals_ext3` vs CPU
//! `FriLayerMerkleTree::build` (PairKeccak256 backend over ext3 pairs).

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::traits::ByteConversion;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha3::{Digest, Keccak256};

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

fn cpu_hash_pair_bytes(a: &Fp3, b: &Fp3) -> [u8; 32] {
    let mut buf = [0u8; 48];
    a.write_bytes_be(&mut buf[0..24]);
    b.write_bytes_be(&mut buf[24..48]);
    let mut h = Keccak256::new();
    h.update(&buf);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

fn cpu_hash_pair_nodes(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(left);
    h.update(right);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

fn cpu_fri_layer_nodes(evals: &[Fp3]) -> Vec<[u8; 32]> {
    let num_leaves = evals.len() / 2;
    assert!(num_leaves.is_power_of_two() && num_leaves >= 1);
    let total = 2 * num_leaves - 1;
    let mut nodes: Vec<[u8; 32]> = vec![[0u8; 32]; total];
    for j in 0..num_leaves {
        nodes[num_leaves - 1 + j] = cpu_hash_pair_bytes(&evals[2 * j], &evals[2 * j + 1]);
    }
    let mut level_begin = num_leaves - 1;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        for k in 0..n_pairs {
            let l = nodes[level_begin + 2 * k];
            let r = nodes[level_begin + 2 * k + 1];
            nodes[new_begin + k] = cpu_hash_pair_nodes(&l, &r);
        }
        level_begin = new_begin;
    }
    nodes
}

fn run_parity(log_num_leaves: u32, seed: u64) {
    let num_leaves = 1usize << log_num_leaves;
    let num_evals = num_leaves * 2;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evals: Vec<Fp3> = (0..num_evals).map(|_| rand_ext3(&mut rng)).collect();
    let evals_u64 = ext3_to_u64s(&evals);

    let cpu_nodes = cpu_fri_layer_nodes(&evals);
    let gpu_bytes = math_cuda::merkle::build_fri_layer_tree_from_evals_ext3(&evals_u64).unwrap();

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
