//! The RPX device paths `math_cuda::rpx_paths` switches between produce the
//! same bytes, and the benches that measure what each one moves.
//!
//! - The Merkle tops: the half-warp permutation and the walk that uses it for
//!   narrow levels and the tail (`math_cuda::rpx::TreeWalk::WARP`, the
//!   default), against a thread per parent (`TreeWalk::ThreadPerParent`).
//! - The grind: the work-queue kernel on a card-filling grid (the default),
//!   against the stride kernel.
//! - The permutation: the limb multiply and every variant the source
//!   instantiates, against the 64-bit multiply and the host.
//!
//! Every parity test runs both paths in one process through explicit entry
//! points, so none of them depends on the `LAMBDA_VM_RPX_*` switches. The
//! comparisons are byte for byte (nodes, nonces) or raw word for word
//! (permutation states), path against path and, where it is cheap, against the
//! host oracle.
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda --test rpx_device_paths -- --nocapture
//! cargo test -p lambda-vm-prover --release --features cuda --test rpx_device_paths -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "cuda")]

use crypto::merkle_tree::merkle::MerkleTree;
use lambda_vm_prover::lfm::algebraic_commit::{AlgebraicPairBackend, RpxCommit, RpxStarkHash};
use lambda_vm_prover::lfm::hash::LfmHasher;
use lambda_vm_prover::lfm::rpx::Rpx256;
use lambda_vm_prover::tables::types::FE;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math_cuda::grinding::Knobs;
use math_cuda::rpx::TreeWalk;

type Ext3 = Degree3GoldilocksExtensionField;
type Fp3 = FieldElement<Ext3>;
type RpxGrind = stark::config::GrindingDigest<RpxStarkHash>;

const P: u64 = 0xFFFF_FFFF_0000_0001;

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn canonical(v: u64) -> u64 {
    if v >= P { v - P } else { v }
}

/// Random states plus the edges; every fifth lane a raw twin `c + p`.
fn probe_states(seed: u64, n: usize) -> Vec<[u64; 12]> {
    let mut rng = SplitMix(seed);
    let mut states = vec![[0u64; 12], [P - 1; 12], [u64::MAX; 12]];
    for k in 0..n {
        states.push(core::array::from_fn(|i| {
            if (k + i) % 5 == 0 {
                rng.next() % 0xFFFF_FFFF + P
            } else {
                rng.next() % P
            }
        }));
    }
    states
}

fn host_permute(s: &[u64; 12]) -> [u64; 12] {
    let out = Rpx256.permute(core::array::from_fn(|i| FE::from(s[i])));
    core::array::from_fn(|i| canonical(*out[i].value()))
}

// ===========================================================================
// The permutation: the limb multiply and the variants.
// ===========================================================================

/// The device PTX of `mul_limb` / `sqr_limb` against 128-bit arithmetic: the
/// words are congruent to the product (they need not be canonical).
#[test]
fn limb_primitives_match_128_bit_arithmetic_on_the_device() {
    let edges = [
        0u64,
        1,
        2,
        P - 1,
        P,
        P + 1,
        0xFFFF_FFFF,
        0x1_0000_0000,
        1 << 63,
        u64::MAX,
        u64::MAX - 1,
        0xFFFF_FFFF_0000_0000,
    ];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &x in &edges {
        for &y in &edges {
            a.push(x);
            b.push(y);
        }
    }
    let mut rng = SplitMix(0x11B);
    for _ in 0..(1 << 16) {
        a.push(rng.next());
        b.push(rng.next());
    }
    let got = math_cuda::rpx::limb_probe(&a, &b).expect("limb probe (needs a GPU)");
    for (i, ((&x, &y), g)) in a.iter().zip(&b).zip(&got).enumerate() {
        let want_mul = ((x as u128 * y as u128) % P as u128) as u64;
        let want_sqr = ((x as u128 * x as u128) % P as u128) as u64;
        assert_eq!(canonical(g[0]), want_mul, "mul_limb at #{i}: {x} * {y}");
        assert_eq!(canonical(g[1]), want_sqr, "sqr_limb at #{i}: {x}^2");
        assert_eq!(
            canonical(g[2]),
            want_mul,
            "goldilocks::mul at #{i} (the control)"
        );
    }
}

/// Every compiled variant reproduces the host oracle, raw, one permutation and
/// three chained.
#[test]
fn every_permutation_variant_matches_the_host_oracle() {
    let states = probe_states(0x0052_5058, 509);
    let variants = math_cuda::rpx::chain_probe_variants().expect("variants (needs a GPU)");
    assert_eq!(variants, vec![0, 1, 3, 4, 5, 7, 9], "the probe set");
    let once: Vec<[u64; 12]> = states.iter().map(host_permute).collect();
    let thrice: Vec<[u64; 12]> = states
        .iter()
        .map(|s| host_permute(&host_permute(&host_permute(s))))
        .collect();
    for &v in &variants {
        let got1 = math_cuda::rpx::permute_chain_probe(v, &states, 1).expect("chain probe");
        let got3 = math_cuda::rpx::permute_chain_probe(v, &states, 3).expect("chain probe");
        for n in 0..states.len() {
            assert_eq!(got1[n], once[n], "variant {v}, state {n}, one permutation");
            assert_eq!(got3[n], thrice[n], "variant {v}, state {n}, three chained");
        }
    }
    // The module's own permutation (this cubin's variant) through `rpx_permute_probe`.
    let module = math_cuda::rpx::permute_probe(&states).expect("permute probe");
    assert_eq!(module, once, "rpx_permute_probe must match the oracle");
}

// ===========================================================================
// The Merkle tops: the half-warp permutation and the warp tree walk.
// ===========================================================================

#[test]
fn the_half_warp_permutation_matches_the_host_oracle() {
    // An odd count, so the last warp has a dead half.
    let states = probe_states(0x5117, 1001);
    let got = math_cuda::rpx::permute_warp_probe(&states).expect("warp probe (needs a GPU)");
    for (n, (s, g)) in states.iter().zip(&got).enumerate() {
        assert_eq!(*g, host_permute(s), "state {n}");
    }
}

fn random_leaves(num_leaves: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix(seed);
    (0..num_leaves * 32).map(|_| rng.next() as u8).collect()
}

/// Every warp walk builds the same tree as a thread per parent, node for node,
/// from 2 to 2^18 leaves — which covers every regime: tail only, warp levels
/// into the tail, and wide per-parent levels above them. Arbitrary leaf bytes are fair here:
/// both walks decode big-endian words without reducing them.
#[test]
fn warp_walks_match_the_thread_per_parent_walk() {
    let walks = [
        TreeWalk::WARP,
        // Every narrow level through the warp level kernel, and a short tail.
        TreeWalk::Warp {
            level_max_pairs: u64::MAX,
            tail_max_pairs: 2,
        },
        // No warp level at all: per-parent levels straight into the warp tail.
        TreeWalk::Warp {
            level_max_pairs: 0,
            tail_max_pairs: 64,
        },
    ];
    for log in 1u32..=18 {
        let leaves = random_leaves(1 << log, 0x7EE + log as u64);
        let want =
            math_cuda::rpx::build_merkle_tree_on_device_with(&leaves, TreeWalk::ThreadPerParent)
                .expect("per-parent walk (needs a GPU)");
        for walk in walks {
            let got =
                math_cuda::rpx::build_merkle_tree_on_device_with(&leaves, walk).expect("warp walk");
            assert!(
                got == want,
                "{walk:?} differs from the per-parent walk at 2^{log} leaves"
            );
        }
        // A control that can fail: one flipped bit in the LAST leaf must move
        // the warp walk's root (the last leaf reaches the root through the
        // rightmost pair of every level, the tail's last lane included).
        let mut tampered = leaves.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        let t = math_cuda::rpx::build_merkle_tree_on_device_with(&tampered, TreeWalk::WARP)
            .expect("warp walk");
        assert_ne!(
            t[..32],
            want[..32],
            "a flipped leaf bit must move the root at 2^{log}"
        );
    }
}

/// The warp walk against the HOST tree (the Pair backend's FRI-layer tree),
/// node for node — the oracle the per-parent walk is pinned to elsewhere.
#[test]
fn the_warp_walk_matches_the_host_tree() {
    for log in [1u32, 3, 6, 7, 8, 10, 13, 15] {
        let num_leaves = 1usize << log;
        let mut rng = SplitMix(0xABC + log as u64);
        let evals: Vec<Fp3> = (0..num_leaves * 2)
            .map(|_| {
                Fp3::new([
                    FieldElement::from(rng.next() % P),
                    FieldElement::from(rng.next() % P),
                    FieldElement::from(rng.next() % P),
                ])
            })
            .collect();
        let leaves: Vec<[Fp3; 2]> = evals.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        let cpu_tree =
            MerkleTree::<AlgebraicPairBackend<Ext3, RpxCommit>>::build(&leaves).expect("CPU tree");
        let cpu_nodes = cpu_tree.nodes();
        let leaf_bytes: Vec<u8> = cpu_nodes[num_leaves - 1..]
            .iter()
            .flat_map(|n| n.iter().copied())
            .collect();
        let got = math_cuda::rpx::build_merkle_tree_on_device_with(&leaf_bytes, TreeWalk::WARP)
            .expect("warp walk (needs a GPU)");
        assert_eq!(got.len(), cpu_nodes.len() * 32, "node count at 2^{log}");
        for (i, want) in cpu_nodes.iter().enumerate() {
            assert_eq!(
                &got[i * 32..(i + 1) * 32],
                &want[..],
                "node {i} at 2^{log} leaves"
            );
        }
    }
}

// ===========================================================================
// The grind: the work-queue kernel.
// ===========================================================================

fn seed_for(i: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&(i as u64).to_le_bytes());
    seed[8] = 0xA5;
    seed
}

/// The queue returns the stride kernel's nonce for 256 seeds at the production
/// factor, at the card-filling grid, at the stride kernel's grid and a small one,
/// and at scan factor 1 (where some seeds miss their first block and relaunch).
#[test]
fn the_queue_grind_returns_the_stride_grinds_nonce() {
    let factor = 20u8;
    let fill = math_cuda::grinding::queue_grid().expect("queue grid (needs a GPU)");
    let scan1 = Knobs {
        scan: 1,
        grid: Knobs::DEFAULT.grid,
    };
    let mut relaunched = 0usize;
    for i in 0..256 {
        let seed = seed_for(i);
        let felts = stark::grinding::inner_hash_felts::<RpxGrind>(&seed, factor);
        let want = math_cuda::grinding::generate_nonce_rpx_gpu_at(&felts, factor, Knobs::DEFAULT)
            .expect("stride grind (needs a GPU)");
        for grid in [fill, 1024, 170] {
            let got = math_cuda::grinding::generate_nonce_rpx_gpu_queue_at(
                &felts,
                factor,
                Knobs::DEFAULT,
                grid,
            )
            .expect("queue grind");
            assert_eq!(got, want, "seed {i}, grid {grid}");
        }
        let got = math_cuda::grinding::generate_nonce_rpx_gpu_queue_at(&felts, factor, scan1, fill)
            .expect("queue grind at scan 1");
        assert_eq!(got, want, "seed {i}, scan 1");
        if want >= 1 << factor {
            relaunched += 1;
        }
    }
    assert!(
        relaunched > 0,
        "scan 1 must have exercised a relaunch on some seed"
    );
    println!(
        "queue parity: 256 seeds x 4 queue arms identical; {relaunched} seeds needed a relaunch at scan 1"
    );
}

/// Small factors against the HOST's smallest valid nonce.
#[test]
fn the_queue_grind_returns_the_host_smallest_nonce() {
    let fill = math_cuda::grinding::queue_grid().expect("queue grid (needs a GPU)");
    for factor in [12u8, 13, 14, 16] {
        for i in 0..16 {
            let seed = seed_for(1000 + i);
            let felts = stark::grinding::inner_hash_felts::<RpxGrind>(&seed, factor);
            let got = math_cuda::grinding::generate_nonce_rpx_gpu_queue_at(
                &felts,
                factor,
                Knobs::DEFAULT,
                fill,
            )
            .expect("queue grind");
            let want = crypto::grinding::generate_nonce_smallest::<RpxGrind>(&seed, factor)
                .expect("host smallest nonce");
            assert_eq!(got, want, "factor {factor}, seed {i}");
            assert!(stark::grinding::is_valid_nonce::<RpxGrind>(
                &seed, got, factor
            ));
        }
    }
}
