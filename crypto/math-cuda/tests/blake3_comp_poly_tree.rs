//! Parity: the device BLAKE3 composition-polynomial tree must equal the CPU
//! tree node for node, through BOTH wrappers that build it.
//!
//! `build_comp_poly_tree_from_evals_ext3_keep` takes host-side interleaved parts
//! and stages them through the pinned de-interleave buffer;
//! `build_comp_poly_tree_from_slabs_dev` takes an already-resident slab buffer
//! and never touches the host. They share the leaf kernel and the level walk but
//! not the staging, so a de-interleave bug shows in the first and not the second
//! — which is why both are exercised here rather than only the one the leaf test
//! happens to call.
//!
//! CPU reference is the production leaf function plus the production tree
//! builder, so nothing in the comparison is written for the test.
//!
//! Needs a GPU.

mod blake3_reference;

use blake3_reference::expected_device_rounds;
use crypto::hash::blake3::BLAKE3_ROUNDS;
use crypto::merkle_tree::backends::types::BatchBlake3Backend;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::commitment::leaves_bit_reversed_grouped;

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

fn assert_lockstep() {
    assert_eq!(
        BLAKE3_ROUNDS,
        expected_device_rounds(),
        "crypto's blake3-6round and math-cuda's are out of lockstep"
    );
}

/// The CPU node buffer for these parts: production row-pair leaves, production
/// tree walk.
fn cpu_nodes(parts: &[Vec<Fp3>]) -> Vec<[u8; 32]> {
    let leaves = leaves_bit_reversed_grouped::<Ext3, BatchBlake3Backend<Ext3>>(parts, 2);
    let tree = MerkleTree::<BatchBlake3Backend<Ext3>>::build_from_hashed_leaves(leaves).unwrap();
    tree.nodes().to_vec()
}

fn interleave(parts: &[Vec<Fp3>], lde_size: usize) -> Vec<Vec<u64>> {
    parts
        .iter()
        .map(|p| {
            let mut v = vec![0u64; 3 * lde_size];
            for (i, e) in p.iter().enumerate() {
                v[i * 3] = *e.value()[0].value();
                v[i * 3 + 1] = *e.value()[1].value();
                v[i * 3 + 2] = *e.value()[2].value();
            }
            v
        })
        .collect()
}

/// The de-interleaved slab layout the device wrapper consumes directly:
/// component `k` of part `c` at `(c*3 + k) * lde_size`.
fn slabs(parts: &[Vec<Fp3>], lde_size: usize) -> Vec<u64> {
    let mut buf = vec![0u64; 3 * parts.len() * lde_size];
    for (c, p) in parts.iter().enumerate() {
        for (r, e) in p.iter().enumerate() {
            buf[(c * 3) * lde_size + r] = *e.value()[0].value();
            buf[(c * 3 + 1) * lde_size + r] = *e.value()[1].value();
            buf[(c * 3 + 2) * lde_size + r] = *e.value()[2].value();
        }
    }
    buf
}

fn assert_nodes_eq(gpu: &[u8], cpu: &[[u8; 32]], what: &str) {
    assert_eq!(gpu.len(), cpu.len() * 32, "{what}: node count");
    for (i, expected) in cpu.iter().enumerate() {
        assert_eq!(
            &gpu[i * 32..(i + 1) * 32],
            &expected[..],
            "{what}: node {i} mismatch"
        );
    }
}

fn run_parity(log_lde: u32, num_parts: usize, seed: u64) {
    assert_lockstep();
    let lde_size = 1usize << log_lde;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let parts: Vec<Vec<Fp3>> = (0..num_parts)
        .map(|_| (0..lde_size).map(|_| rand_ext3(&mut rng)).collect())
        .collect();

    let expected = cpu_nodes(&parts);
    let what = format!("log_lde={log_lde} parts={num_parts}");

    let be = math_cuda::device::backend().unwrap();

    // Route 1: host-side interleaved parts through the pinned staging path.
    let interleaved = interleave(&parts, lde_size);
    let slices: Vec<&[u64]> = interleaved.iter().map(|v| v.as_slice()).collect();
    let keep = math_cuda::blake3::build_comp_poly_tree_from_evals_ext3_keep(&slices).unwrap();
    {
        let stream = be.next_stream();
        let nodes: Vec<u8> = stream.clone_dtoh(&*keep.nodes).unwrap();
        assert_nodes_eq(&nodes, &expected, &format!("keep {what}"));
        assert_eq!(&keep.root[..], &expected[0][..], "keep {what}: root");
        assert_eq!(keep.leaves_len, lde_size / 2, "keep {what}: leaf count");
    }

    // Route 2: an already-resident slab buffer, no host staging.
    {
        let stream = be.next_stream();
        let buf = stream.clone_htod(&slabs(&parts, lde_size)).unwrap();
        stream.synchronize().unwrap();
        let dev = math_cuda::blake3::build_comp_poly_tree_from_slabs_dev(
            &stream, &buf, num_parts, lde_size,
        )
        .unwrap();
        let nodes: Vec<u8> = stream.clone_dtoh(&*dev.nodes).unwrap();
        assert_nodes_eq(&nodes, &expected, &format!("slabs {what}"));
        assert_eq!(&dev.root[..], &expected[0][..], "slabs {what}: root");
    }
}

/// Small trees: the tail kernel builds every level in one launch.
#[test]
fn blake3_comp_poly_tree_small() {
    for log_lde in [2u32, 4, 6, 8] {
        for num_parts in [1usize, 2, 5] {
            run_parity(log_lde, num_parts, 300 + log_lde as u64 + num_parts as u64);
        }
    }
}

/// Deep enough to cross from the per-level kernel into the tail.
#[test]
fn blake3_comp_poly_tree_medium() {
    for log_lde in [10u32, 12, 14] {
        run_parity(log_lde, 17, 700 + log_lde as u64);
    }
}

#[test]
fn blake3_comp_poly_tree_large() {
    run_parity(18, 3, 4242);
}
