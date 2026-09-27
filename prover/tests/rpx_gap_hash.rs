//! ⚠ GAP campaign, HASH lane (K3, K4, K5): the new RPX device paths produce the
//! shipped paths' bytes, and the benches that measure what each one moves.
//!
//! - K3: the half-warp permutation and the Merkle walk that uses it for narrow
//!   levels and the tail (`math_cuda::rpx::TreeWalk::WARP`).
//! - K4: the work-queue grind on a card-filling grid.
//! - K5: the permutation's limb multiply and the compiled variants.
//!
//! Every parity test runs both paths in one process through explicit entry
//! points, so none of them depends on the `LAMBDA_VM_GAP_*` environment. The
//! comparisons are byte for byte (nodes, nonces) or raw word for word
//! (permutation states), against the shipped device path and, where it is cheap,
//! against the host oracle.
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda --test rpx_gap_hash -- --nocapture
//! cargo test -p lambda-vm-prover --release --features cuda --test rpx_gap_hash -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "cuda")]

use std::time::Instant;

use crypto::merkle_tree::merkle::MerkleTree;
use lambda_vm_prover::lfm::algebraic_commit::{AlgebraicPairBackend, RpxCommit, RpxStarkHash};
use lambda_vm_prover::lfm::hash::LfmHasher;
use lambda_vm_prover::lfm::rpx::Rpx256;
use lambda_vm_prover::tables::types::FE;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math_cuda::grinding::Knobs;
use math_cuda::rpx::{LevelKernel, TreeWalk};

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
// K5 — the limb multiply and the permutation variants.
// ===========================================================================

/// The device PTX of `mul_limb` / `sqr_limb` against 128-bit arithmetic: the
/// words are congruent to the product (they need not be canonical).
#[test]
fn k5_limb_primitives_match_128_bit_arithmetic_on_the_device() {
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
fn k5_every_permutation_variant_matches_the_host_oracle() {
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
    // The module's own permutation (this cubin's variant) through the shipped probe.
    let shipped = math_cuda::rpx::permute_probe(&states).expect("permute probe");
    assert_eq!(shipped, once, "rpx_permute_probe must match the oracle");
}

// ===========================================================================
// K3 — the half-warp permutation and the warp tree walk.
// ===========================================================================

#[test]
fn k3_warp_permutation_matches_the_host_oracle() {
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

/// Every walk builds the same tree as the shipped walk, node for node, from 2
/// to 2^18 leaves — which covers every regime: tail only, warp levels into the
/// tail, and shipped wide levels above them. Arbitrary leaf bytes are fair here:
/// both walks decode big-endian words without reducing them.
#[test]
fn k3_warp_walks_match_the_shipped_walk() {
    let walks = [
        TreeWalk::WARP,
        // Every narrow level through the warp level kernel, and a short tail.
        TreeWalk::Warp {
            level_max_pairs: u64::MAX,
            tail_max_pairs: 2,
        },
        // No warp level at all: shipped levels straight into the warp tail.
        TreeWalk::Warp {
            level_max_pairs: 0,
            tail_max_pairs: 64,
        },
    ];
    for log in 1u32..=18 {
        let leaves = random_leaves(1 << log, 0x7EE + log as u64);
        let want = math_cuda::rpx::build_merkle_tree_on_device_with(&leaves, TreeWalk::Shipped)
            .expect("shipped walk (needs a GPU)");
        for walk in walks {
            let got =
                math_cuda::rpx::build_merkle_tree_on_device_with(&leaves, walk).expect("warp walk");
            assert!(
                got == want,
                "{walk:?} differs from the shipped walk at 2^{log} leaves"
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
/// node for node — the oracle the shipped walk is pinned to elsewhere.
#[test]
fn k3_warp_walk_matches_the_host_tree() {
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
// K4 — the work-queue grind.
// ===========================================================================

fn seed_for(i: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&(i as u64).to_le_bytes());
    seed[8] = 0xA5;
    seed
}

/// The queue returns the shipped kernel's nonce for 256 seeds at the production
/// factor, at the card-filling grid, at the shipped grid and at a small one,
/// and at scan factor 1 (where some seeds miss their first block and relaunch).
#[test]
fn k4_queue_grind_returns_the_shipped_nonce() {
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
            .expect("shipped grind (needs a GPU)");
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
        "K4 parity: 256 seeds x 4 queue arms identical; {relaunched} seeds needed a relaunch at scan 1"
    );
}

/// Small factors against the HOST's smallest valid nonce.
#[test]
fn k4_queue_grind_returns_the_host_smallest_nonce() {
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

// ===========================================================================
// Benches (`--ignored --nocapture --test-threads=1`, on an idle card).
// ===========================================================================

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// ★ K4 MECHANISM: permutations EXECUTED and milliseconds per grind, shipped vs
/// queue, same 256 seeds, arms interleaved per seed. Pre-registered (the
/// brief's note, before the run): executed ratio queue/shipped 0.68–0.80, ms
/// ratio 0.60–0.80, nonces identical 256/256.
#[test]
#[ignore = "bench: run on an idle GPU with --ignored --nocapture --test-threads=1"]
fn k4_bench_grind_executed_and_time() {
    let factor = 20u8;
    let fill = math_cuda::grinding::queue_grid().expect("queue grid (needs a GPU)");
    let posture = Knobs::DEFAULT;
    let stride = posture.stride(math_cuda::grinding::RPX_BLOCK_DIM);
    // Warm-up of all four kernels, excluded.
    let w = stark::grinding::inner_hash_felts::<RpxGrind>(&seed_for(9999), factor);
    let _ = math_cuda::grinding::generate_nonce_rpx_gpu_at(&w, factor, posture);
    let _ = math_cuda::grinding::search_counted(&w, factor, posture, 1);
    let _ = math_cuda::grinding::generate_nonce_rpx_gpu_queue_at(&w, factor, posture, fill);
    let _ = math_cuda::grinding::search_queue_counted(&w, factor, posture, fill);

    let (mut ms_ship, mut ms_queue) = (Vec::new(), Vec::new());
    let (mut ex_ship, mut ex_queue, mut nonces) = (0u128, 0u128, 0u128);
    for i in 0..256 {
        let felts = stark::grinding::inner_hash_felts::<RpxGrind>(&seed_for(i), factor);
        let time = |queue: bool| -> (u64, f64) {
            let t = Instant::now();
            let n = if queue {
                math_cuda::grinding::generate_nonce_rpx_gpu_queue_at(&felts, factor, posture, fill)
            } else {
                math_cuda::grinding::generate_nonce_rpx_gpu_at(&felts, factor, posture)
            }
            .expect("grind");
            (n, t.elapsed().as_secs_f64() * 1e3)
        };
        // ABBA within the seed.
        let (a1, ta1) = time(false);
        let (b1, tb1) = time(true);
        let (b2, tb2) = time(true);
        let (a2, ta2) = time(false);
        assert!(a1 == b1 && b1 == b2 && b2 == a2, "seed {i}: nonces differ");
        ms_ship.push((ta1 + ta2) / 2.0);
        ms_queue.push((tb1 + tb2) / 2.0);
        let cs = math_cuda::grinding::search_counted(&felts, factor, posture, 1).expect("counted");
        let cq = math_cuda::grinding::search_queue_counted(&felts, factor, posture, fill)
            .expect("counted");
        assert!(
            cs.nonce == a1 && cq.nonce == a1,
            "seed {i}: counted nonces differ"
        );
        ex_ship += cs.executed as u128;
        ex_queue += cq.executed as u128;
        nonces += a1 as u128;
    }
    let n = 256.0;
    let (mean_ship, mean_queue) = (
        ms_ship.iter().sum::<f64>() / n,
        ms_queue.iter().sum::<f64>() / n,
    );
    println!(
        "★ K4 GRIND BENCH (factor {factor}, scan {}, 256 seeds, per-seed ABBA)",
        posture.scan
    );
    println!(
        "  shipped: grid {} (stride {stride}); queue: grid {fill} (stride {})",
        posture.grid,
        fill as u64 * math_cuda::grinding::RPX_BLOCK_DIM as u64
    );
    println!(
        "  mean nonce {:.0}; executed/seed shipped {:.0} = nonce + {:.2} strides; queue {:.0} = nonce + {:.2} x its grid's threads",
        nonces as f64 / n,
        ex_ship as f64 / n,
        (ex_ship as f64 - nonces as f64) / n / stride as f64,
        ex_queue as f64 / n,
        (ex_queue as f64 - nonces as f64) / n / (fill as f64 * 128.0),
    );
    println!(
        "  EXECUTED ratio queue/shipped {:.4}  (pre-registered 0.68-0.80)",
        ex_queue as f64 / ex_ship as f64
    );
    println!(
        "  ms/grind mean shipped {mean_ship:.3} queue {mean_queue:.3}; median shipped {:.3} queue {:.3}; MS ratio (means) {:.4}  (pre-registered 0.60-0.80)",
        median(&mut ms_ship.clone()),
        median(&mut ms_queue.clone()),
        mean_queue / mean_ship
    );
}

/// ★ K3 MECHANISM: seconds per launch of each level kernel by width, and per
/// whole inner-tree walk by tree size. Pre-registered: a level of ≤ 16,384
/// pairs costs the shipped kernel ≈ 110–125 µs (one single-thread chain) and
/// the warp kernel ≤ 70 µs at 16,384 pairs and ≤ 20 µs at ≤ 2,048; the tree
/// TOP (levels ≤ 16,384 pairs, tail included) drops from ≈ 1.6–1.8 ms to
/// 0.2–0.5 ms per tree.
#[test]
#[ignore = "bench: run on an idle GPU with --ignored --nocapture --test-threads=1"]
fn k3_bench_levels_and_trees() {
    println!("★ K3 LEVEL BENCH: µs per launch (200 back-to-back launches, warm-up excluded)");
    println!(
        "{:>10} {:>12} {:>12} {:>8}",
        "pairs", "shipped", "warp", "ratio"
    );
    let mut p = 128u64;
    while p <= 1 << 17 {
        let s =
            math_cuda::rpx::level_bench(LevelKernel::Shipped, p, 200).expect("bench (needs a GPU)");
        let w = math_cuda::rpx::level_bench(LevelKernel::Warp, p, 200).expect("bench");
        println!("{p:>10} {:>12.2} {:>12.2} {:>8.3}", s * 1e6, w * 1e6, w / s);
        p *= 2;
    }
    let st = math_cuda::rpx::level_bench(LevelKernel::ShippedTail, 128, 200).expect("bench");
    let wt = math_cuda::rpx::level_bench(LevelKernel::WarpTail, 64, 200).expect("bench");
    let wl = math_cuda::rpx::level_bench(LevelKernel::Warp, 128, 200).expect("bench");
    println!(
        "  tails: shipped 128 pairs -> root {:.2} µs; warp level 128 + warp tail 64 -> root {:.2} µs",
        st * 1e6,
        (wl + wt) * 1e6
    );

    println!("★ K3 TREE BENCH: µs per inner-tree walk (median of 5 rounds x 20 walks)");
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "leaves", "shipped", "warp", "delta"
    );
    for log in [8u32, 12, 15, 16, 18, 20, 21, 22] {
        let (mut s, mut w) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            s.push(math_cuda::rpx::tree_bench(1 << log, TreeWalk::Shipped, 20).expect("bench"));
            w.push(math_cuda::rpx::tree_bench(1 << log, TreeWalk::WARP, 20).expect("bench"));
        }
        let (s, w) = (median(&mut s), median(&mut w));
        println!(
            "{:>8} {:>12.1} {:>12.1} {:>10.1}",
            format!("2^{log}"),
            s * 1e6,
            w * 1e6,
            (w - s) * 1e6
        );
    }
    println!("★ K3 THRESHOLD SWEEP at 2^22 leaves (level_max_pairs; tail 64)");
    for level_max_pairs in [2048u64, 4096, 8192, 16_384, 32_768, 65_536] {
        let walk = TreeWalk::Warp {
            level_max_pairs,
            tail_max_pairs: 64,
        };
        let mut t = Vec::new();
        for _ in 0..5 {
            t.push(math_cuda::rpx::tree_bench(1 << 22, walk, 10).expect("bench"));
        }
        println!(
            "  level_max_pairs {level_max_pairs:>6}: {:.1} µs",
            median(&mut t) * 1e6
        );
    }
}

/// ★ K5 MECHANISM: nanoseconds per permutation of each compiled variant, on
/// the chained probe (4 full waves of that variant's own occupancy x 16
/// permutations x 16 launches — variants differ in registers, so a fixed launch
/// size would give each a different wave tail), variants interleaved over 7
/// rounds, median reported. Pre-registered: variant 1 (limb multiply) at
/// 0.75–1.00 of variant 0; variant 3 against 1 within ±10% (the sign says
/// whether the multiply pipe or issue binds); the unrolled variants (4, 5, 7,
/// 9) at 0.88–1.02 of their rolled twins.
#[test]
#[ignore = "bench: run on an idle GPU with --ignored --nocapture --test-threads=1"]
fn k5_bench_permutation_variants() {
    let variants = math_cuda::rpx::chain_probe_variants().expect("variants (needs a GPU)");
    let (k, iters) = (16u64, 16usize);
    let sizes: Vec<usize> = variants
        .iter()
        .map(|&v| 4 * math_cuda::rpx::chain_probe_resident_threads(v).expect("occupancy") as usize)
        .collect();
    for (i, &v) in variants.iter().enumerate() {
        let _ = math_cuda::rpx::permute_chain_bench(v, sizes[i], k, 2).expect("warm-up");
    }
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut shape: Vec<(u32, u32)> = vec![(0, 0); variants.len()];
    for round in 0..7 {
        for j in 0..variants.len() {
            let idx = (j + round) % variants.len();
            let b = math_cuda::rpx::permute_chain_bench(variants[idx], sizes[idx], k, iters)
                .expect("bench");
            per[idx].push(b.ns_per_perm());
            shape[idx] = (b.regs, b.blocks_per_sm);
        }
    }
    let base = median(&mut per[0].clone());
    println!(
        "★ K5 PERMUTATION BENCH: ns per permutation, whole card (4 full waves x 16 chained x 16 launches, median of 7)"
    );
    println!(
        "{:>8} {:>10} {:>8} {:>6} {:>10} {:>9}",
        "variant", "ns/perm", "vs v0", "regs", "blocks/SM", "threads"
    );
    for (i, &v) in variants.iter().enumerate() {
        let m = median(&mut per[i].clone());
        println!(
            "{v:>8} {m:>10.4} {:>8.4} {:>6} {:>10} {:>9}",
            m / base,
            shape[i].0,
            shape[i].1,
            sizes[i]
        );
    }
    println!(
        "  this process's rpx module: {}",
        match math_cuda::gap_hash::knobs().k5 {
            Some(v) => format!("variant {v} (LAMBDA_VM_GAP_K5={v})"),
            None => "variant 0 (LAMBDA_VM_GAP_K5 off)".to_string(),
        }
    );
}
