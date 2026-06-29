//! Parity: GPU `gather_merkle_paths_dev` must produce, for each leaf position,
//! the exact `merkle_path` the CPU `MerkleTree::get_proof_by_pos` returns: the
//! same sibling order from leaf to root, byte for byte. This is the gate for
//! gathering R4 query openings on device instead of copying the whole tree.

use crypto::merkle_tree::backends::field_element_vector::FieldElementVectorBackend;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha3::Keccak256;

type CpuTree = MerkleTree<FieldElementVectorBackend<GoldilocksField, Keccak256, 32>>;

fn run_gather_parity(log_n: u32, seed: u64) {
    let leaves_len = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let leaves: Vec<[u8; 32]> = (0..leaves_len)
        .map(|_| {
            let mut arr = [0u8; 32];
            rng.fill(&mut arr[..]);
            arr
        })
        .collect();

    let mut flat = Vec::with_capacity(leaves_len * 32);
    for l in &leaves {
        flat.extend_from_slice(l);
    }

    // Build the tree on device, then upload its nodes back as the resident
    // buffer the gather reads (build_merkle_tree_on_device returns host bytes).
    let gpu_nodes_bytes = math_cuda::merkle::build_merkle_tree_on_device(&flat).unwrap();

    // CPU reference tree over the same backend as the prover.
    let cpu_tree = CpuTree::build_from_hashed_leaves(leaves).unwrap();

    // Query a spread of positions: first, last, and random interior ones.
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
            let g = &paths[(q * depth + level) * 32..(q * depth + level + 1) * 32];
            assert_eq!(
                g,
                &cpu_node[..],
                "path node mismatch: log_n={log_n} pos={pos} level={level}"
            );
        }
    }
}

#[test]
fn merkle_gather_small() {
    for log_n in 1u32..=6 {
        run_gather_parity(log_n, 200 + log_n as u64);
    }
}

#[test]
fn merkle_gather_large() {
    run_gather_parity(18, 7777);
}
