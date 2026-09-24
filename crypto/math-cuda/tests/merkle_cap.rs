//! Parity: `read_cap_dev` must return, for every cap height, exactly the nodes
//! the host `MerkleTree::cap` returns — the `2^c` nodes `c` levels below the
//! root, left to right, byte for byte. This is the gate for reading a
//! device-resident tree's Merkle cap in the STARK R4 cap post-pass
//! (design/CAP.md §4.2) instead of copying the whole tree.

use crypto::merkle_tree::backends::field_element_vector::FieldElementVectorBackend;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha3::Keccak256;

type CpuTree = MerkleTree<FieldElementVectorBackend<GoldilocksField, Keccak256, 32>>;

fn random_leaves(leaves_len: usize, seed: u64) -> Vec<[u8; 32]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..leaves_len)
        .map(|_| {
            let mut arr = [0u8; 32];
            rng.fill(&mut arr[..]);
            arr
        })
        .collect()
}

fn flat(leaves: &[[u8; 32]]) -> Vec<u8> {
    leaves.iter().flat_map(|l| l.iter().copied()).collect()
}

/// Every height `c <= min(depth, 6)` of a keccak tree with `2^log_n` leaves:
/// the device read equals the host cap, and the heap slice of the device's own
/// node buffer.
fn keccak_cap_parity(log_n: u32, seed: u64) {
    let leaves_len = 1usize << log_n;
    let leaves = random_leaves(leaves_len, seed);
    let gpu_nodes = math_cuda::merkle::build_merkle_tree_on_device(&flat(&leaves)).unwrap();
    let cpu_tree = CpuTree::build_from_hashed_leaves(leaves).unwrap();

    let be = math_cuda::device::backend().unwrap();
    let stream = be.next_stream();
    let nodes_dev = stream.clone_htod(&gpu_nodes).unwrap();
    stream.synchronize().unwrap();

    let depth = log_n as usize;
    for c in 0..=depth.min(6) {
        let got = math_cuda::merkle::read_cap_dev(&nodes_dev, leaves_len, c, &stream).unwrap();
        let want: Vec<u8> = cpu_tree
            .cap(c)
            .unwrap()
            .iter()
            .flat_map(|n| n.iter().copied())
            .collect();
        assert_eq!(got.len(), (1 << c) * 32, "log_n={log_n} c={c}");
        assert_eq!(got, want, "keccak cap mismatch: log_n={log_n} c={c}");
        assert_eq!(
            got,
            gpu_nodes[((1 << c) - 1) * 32..((2 << c) - 1) * 32],
            "log_n={log_n} c={c}: not the heap slice"
        );
    }
    let root = math_cuda::merkle::read_cap_dev(&nodes_dev, leaves_len, 0, &stream).unwrap();
    assert_eq!(root, cpu_tree.root.to_vec(), "c = 0 is the root");
}

#[test]
fn keccak_cap_matches_the_host_cap_small() {
    for log_n in 1u32..=8 {
        keccak_cap_parity(log_n, 300 + log_n as u64);
    }
}

#[test]
fn keccak_cap_matches_the_host_cap_large() {
    for log_n in [12u32, 18, 22] {
        keccak_cap_parity(log_n, 9000 + log_n as u64);
    }
}

/// RPX trees: the read is hash-agnostic (a D2H of the heap slice), so it is
/// pinned against the device builder's own full node buffer, whose layout the
/// existing RPX tree parity tests pin against the host.
#[test]
fn rpx_cap_is_the_heap_slice() {
    for log_n in [1u32, 2, 5, 10, 16] {
        let leaves_len = 1usize << log_n;
        // RPX digests are four canonical Goldilocks limbs; reduce the random
        // bytes below the modulus so the device hashes valid field elements.
        let leaves: Vec<[u8; 32]> = random_leaves(leaves_len, 77 + log_n as u64)
            .into_iter()
            .map(|mut l| {
                for limb in l.chunks_exact_mut(8) {
                    limb[7] &= 0x7f;
                }
                l
            })
            .collect();
        let gpu_nodes = math_cuda::rpx::build_merkle_tree_on_device(&flat(&leaves)).unwrap();
        let be = math_cuda::device::backend().unwrap();
        let stream = be.next_stream();
        let nodes_dev = stream.clone_htod(&gpu_nodes).unwrap();
        stream.synchronize().unwrap();
        let depth = log_n as usize;
        for c in 0..=depth.min(6) {
            let got = math_cuda::merkle::read_cap_dev(&nodes_dev, leaves_len, c, &stream).unwrap();
            assert_eq!(
                got,
                gpu_nodes[((1 << c) - 1) * 32..((2 << c) - 1) * 32],
                "rpx: log_n={log_n} c={c}"
            );
        }
    }
}

/// The resident tree a real R2 commit keeps (`GpuMerkleTree`): the cap read
/// off it equals its full node buffer's heap slice, and `c = 0` its root.
#[test]
fn a_kept_composition_tree_serves_its_cap() {
    let lde_size = 1usize << 12;
    let mut rng = ChaCha8Rng::seed_from_u64(4242);
    let parts: Vec<Vec<u64>> = (0..2)
        .map(|_| {
            (0..3 * lde_size)
                .map(|_| rng.gen_range(0..0xFFFF_FFFF_0000_0001u64))
                .collect()
        })
        .collect();
    let refs: Vec<&[u64]> = parts.iter().map(|p| p.as_slice()).collect();
    let tree = math_cuda::merkle::build_comp_poly_tree_from_evals_ext3_keep(&refs).unwrap();
    let be = math_cuda::device::backend().unwrap();
    let stream = be.next_stream();
    let all = stream.clone_dtoh(tree.nodes.as_ref()).unwrap();
    stream.synchronize().unwrap();
    let depth = tree.leaves_len.trailing_zeros() as usize;
    assert_eq!(tree.leaves_len, lde_size / 2);
    for c in 0..=depth.min(6) {
        let got =
            math_cuda::merkle::read_cap_dev(&tree.nodes, tree.leaves_len, c, &stream).unwrap();
        assert_eq!(got, all[((1 << c) - 1) * 32..((2 << c) - 1) * 32], "c={c}");
    }
    let root = math_cuda::merkle::read_cap_dev(&tree.nodes, tree.leaves_len, 0, &stream).unwrap();
    assert_eq!(root, tree.root.to_vec());
}
