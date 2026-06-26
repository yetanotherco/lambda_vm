//! Parity: GPU hash group-by dedup (`gpu_dedup`) vs a CPU `HashMap` dedup.
//! Output order is arbitrary, so we compare as maps `(key) -> (mu0, mu1)` and
//! also assert the GPU output has no duplicate keys (it actually deduped).
//!
//! `#[ignore]`'d (needs a GPU). Run with:
//!   cargo test -p math-cuda --release --test dedup -- --ignored --nocapture

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn check(n: usize, distinct: u64, dual: bool, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Draw keys from a small distinct range so there are many duplicates.
    let a: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() % distinct).collect();
    let b: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() % distinct).collect();
    let c: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() % 3).collect();
    let sel: Vec<u64> = (0..n)
        .map(|_| if dual { rng.r#gen::<bool>() as u64 } else { 0 })
        .collect();

    // CPU reference: (a,b,c) -> (mu0, mu1).
    let mut want: HashMap<(u64, u64, u64), (u64, u64)> = HashMap::new();
    for i in 0..n {
        let e = want.entry((a[i], b[i], c[i])).or_default();
        if sel[i] != 0 {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }

    let res = math_cuda::trace::gpu_dedup(&a, &b, &c, &sel).unwrap();
    assert_eq!(res.a.len(), want.len(), "n={n} distinct={distinct}: unique count");

    let mut got: HashMap<(u64, u64, u64), (u64, u64)> = HashMap::new();
    for i in 0..res.a.len() {
        let prev = got.insert((res.a[i], res.b[i], res.c[i]), (res.mu0[i], res.mu1[i]));
        assert!(prev.is_none(), "GPU output has a duplicate key (dedup failed)");
    }
    assert_eq!(got, want, "n={n} distinct={distinct} dual={dual}: dedup mismatch");
    println!("dedup OK: n={n} distinct={distinct} dual={dual} -> {} unique", res.a.len());
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn dedup_matches_cpu() {
    check(1, 1, false, 1);
    check(64, 5, false, 2);
    check(10_000, 100, false, 3); // heavy duplication
    check(10_000, 100, true, 4); // dual counters
    check(1_000_000, 1000, true, 5); // scale, dual
    check(4096, 1 << 30, false, 6); // ~all distinct
}
