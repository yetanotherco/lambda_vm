//! Parity: GPU u128 radix-sort + segmented-reduce vs a CPU reference
//! (sort_unstable + HashMap-style count). Three layers of checks:
//!   1. Unique key set matches (as a multiset, since GPU is sorted).
//!   2. Counts match per unique key.
//!   3. Sum of counts equals input length (sanity).

use math_cuda::device::backend;
use math_cuda::multiplicity_sort::multiplicity_count_multifield_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

fn rand_keys(n: usize, key_space: u128, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut hi = Vec::with_capacity(n);
    let mut lo = Vec::with_capacity(n);
    for _ in 0..n {
        let k = rng.r#gen::<u128>() % key_space;
        hi.push((k >> 64) as u64);
        lo.push(k as u64);
    }
    (hi, lo)
}

fn cpu_reference(hi: &[u64], lo: &[u64]) -> BTreeMap<(u64, u64), u64> {
    let mut map = BTreeMap::new();
    for (&h, &l) in hi.iter().zip(lo.iter()) {
        *map.entry((h, l)).or_insert(0u64) += 1;
    }
    map
}

fn run_parity(n: usize, key_space: u128, seed: u64) {
    let (hi, lo) = rand_keys(n, key_space, seed);
    let cpu_ref = cpu_reference(&hi, &lo);

    let be = backend().unwrap();
    let stream = be.next_stream();
    let hi_dev = if n == 0 {
        stream.alloc_zeros::<u64>(0).unwrap()
    } else {
        stream.clone_htod(&hi).unwrap()
    };
    let lo_dev = if n == 0 {
        stream.alloc_zeros::<u64>(0).unwrap()
    } else {
        stream.clone_htod(&lo).unwrap()
    };
    let res = multiplicity_count_multifield_dev(&hi_dev, &lo_dev, n, &stream).unwrap();

    assert_eq!(
        res.num_unique,
        cpu_ref.len(),
        "num_unique mismatch n={n} key_space={key_space} seed={seed}"
    );

    let gpu_hi = stream.clone_dtoh(&res.unique_hi).unwrap();
    let gpu_lo = stream.clone_dtoh(&res.unique_lo).unwrap();
    let gpu_counts = stream.clone_dtoh(&res.counts).unwrap();
    stream.synchronize().unwrap();

    // Build GPU map for comparison.
    let mut gpu_map = BTreeMap::new();
    for ((h, l), c) in gpu_hi.iter().zip(gpu_lo.iter()).zip(gpu_counts.iter()) {
        gpu_map.insert((*h, *l), *c);
    }

    assert_eq!(
        gpu_map, cpu_ref,
        "(unique, count) map mismatch n={n} key_space={key_space} seed={seed}"
    );

    // Sanity: total counts equal n.
    let total: u64 = gpu_counts.iter().sum();
    assert_eq!(total, n as u64, "sum of counts should equal n");

    // Sanity: GPU output is sorted ascending by (hi, lo).
    for i in 1..res.num_unique {
        let prev = (gpu_hi[i - 1], gpu_lo[i - 1]);
        let cur = (gpu_hi[i], gpu_lo[i]);
        assert!(prev < cur, "unique keys not sorted at i={i}: {prev:?} < {cur:?}");
    }
}

#[test]
fn multiplicity_empty() {
    run_parity(0, 16, 0);
}

#[test]
fn multiplicity_single_block_high_dedup() {
    // Small key space → high duplicate rate → exercises segmented reduce.
    run_parity(256, 8, 1);
}

#[test]
fn multiplicity_single_block_low_dedup() {
    // Large key space → most keys unique → exercises sort path.
    run_parity(200, u128::MAX, 2);
}

#[test]
fn multiplicity_two_block() {
    // Just over single-block: forces multi-block scan.
    run_parity(512, 64, 3);
    run_parity(1024, 256, 4);
}

#[test]
fn multiplicity_recursive_scan() {
    // K > 256 → two-level scan recursion.
    run_parity(1 << 14, 1024, 100);
}

#[test]
fn multiplicity_realistic_branch_shape() {
    // Approximates branch.rs at fib_iterative_1M scale: ~100k operations,
    // moderate dedup (typical branch ops repeat across the trace).
    run_parity(100_000, 1024, 9001);
}
