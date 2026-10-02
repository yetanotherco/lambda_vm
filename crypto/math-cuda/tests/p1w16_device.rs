//! D-HASH stage 1: the Poseidon1 width-16 measurement kernels against the Rust
//! host reference (`crypto::hash::poseidon1_w16`), at both multiply variants,
//! and the microbenchmark against the RPX kernels.
//!
//! Needs a GPU (the box); the host half of the arithmetic is pinned per PR by
//! `make test-p1w16-host-kat`.
//!
//!   cargo test -p math-cuda --release --test p1w16_device
//!   P1W16_LOG_LEN=29 cargo test -p math-cuda --release --test p1w16_device \
//!       -- --ignored --nocapture p1w16_microbench

use crypto::hash::poseidon1_w16 as host;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math_cuda::p1w16;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
const P: u64 = 0xFFFF_FFFF_0000_0001;

fn canon(v: &[Fp]) -> Vec<u64> {
    v.iter().map(|f| f.canonical()).collect()
}

fn random_felts(rng: &mut ChaCha8Rng, n: usize) -> Vec<u64> {
    // Raw u64s, some at or above p: the device must accept raw storage.
    (0..n)
        .map(|i| {
            if i % 17 == 0 {
                P + (i as u64 % 7)
            } else {
                rng.r#gen::<u64>() % P
            }
        })
        .collect()
}

fn fp(v: &[u64]) -> Vec<Fp> {
    v.iter().map(|x| Fp::from(*x)).collect()
}

/// The host reference for variant `v`: the circulant instance, or the Cauchy
/// alternative for `c1`.
fn host_permute(v: usize) -> fn([Fp; 16]) -> [Fp; 16] {
    if p1w16::is_cauchy(v) {
        host::permute_cauchy
    } else {
        host::permute
    }
}

fn host_leaf(v: usize) -> fn(&[Fp]) -> host::Digest {
    if p1w16::is_cauchy(v) {
        host::sponge_leaf_cauchy
    } else {
        host::sponge_leaf
    }
}

fn host_node(v: usize) -> fn(&[host::Digest; 4]) -> host::Digest {
    if p1w16::is_cauchy(v) {
        host::compress4_cauchy
    } else {
        host::compress4
    }
}

#[test]
fn the_device_permutation_matches_the_host_reference() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let n = 4099;
    let states = random_felts(&mut rng, n * 16);
    for v in 0..p1w16::VARIANTS.len() {
        let got = p1w16::permute_many(v, &states).expect("device");
        for (i, chunk) in states.chunks(16).enumerate() {
            let s: [Fp; 16] = core::array::from_fn(|k| Fp::from(chunk[k]));
            assert_eq!(
                got[i * 16..(i + 1) * 16],
                canon(&host_permute(v)(s))[..],
                "variant {v} state {i}"
            );
        }
    }
}

#[test]
fn the_device_coset_leaves_match_the_host_sponge() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for (block, ext3) in [
        (64u64, false),
        (16, false),
        (5, false),
        (16, true),
        (4, true),
    ] {
        let per = if ext3 { 3 } else { 1 };
        let num_leaves = 1027u64;
        let cw = random_felts(&mut rng, (num_leaves * block * per) as usize);
        for v in 0..p1w16::VARIANTS.len() {
            let got = p1w16::leaves_coset(v, &cw, block, ext3).expect("device");
            for j in 0..num_leaves {
                let mut felts = Vec::new();
                for t in 0..block {
                    let at = (j + t * num_leaves) as usize;
                    if ext3 {
                        felts.extend_from_slice(&cw[3 * at..3 * at + 3]);
                    } else {
                        felts.push(cw[at]);
                    }
                }
                let want = canon(&host_leaf(v)(&fp(&felts)));
                assert_eq!(
                    got[4 * j as usize..4 * j as usize + 4],
                    want[..],
                    "variant {v} block {block} ext3 {ext3} leaf {j}"
                );
            }
        }
    }
}

#[test]
fn the_device_4ary_level_matches_the_host_node() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let n = 2049;
    let children = random_felts(&mut rng, n * 16);
    for v in 0..p1w16::VARIANTS.len() {
        let got = p1w16::merkle_level4(v, &children).expect("device");
        for i in 0..n {
            let c = &children[i * 16..(i + 1) * 16];
            let kids: [host::Digest; 4] =
                core::array::from_fn(|q| core::array::from_fn(|l| Fp::from(c[4 * q + l])));
            assert_eq!(
                got[4 * i..4 * i + 4],
                canon(&host_node(v)(&kids))[..],
                "variant {v} node {i}"
            );
        }
    }
}

/// The grind returns the smallest nonce the host scan finds, at a factor where
/// the scan is cheap; a limit of zero finds nothing.
#[test]
fn the_device_grind_returns_the_smallest_valid_nonce() {
    let inner = [11u64, 22, 33, 44];
    let limit = 1u64 << (64 - 12);
    for v in 0..p1w16::VARIANTS.len() {
        let leaf = host_leaf(v);
        let head = |nonce: u64| {
            let felts = fp(&[inner[0], inner[1], inner[2], inner[3], nonce]);
            leaf(&felts)[0].canonical()
        };
        let want = (0..1u64 << 20)
            .find(|&n| head(n) < limit)
            .expect("a hit below 2^20");
        for grid in [1u32, 1024] {
            let got = p1w16::grind(v, &inner, limit, 1 << 20, grid).expect("device");
            assert_eq!(got, Some(want), "variant {v} grid {grid}");
        }
        assert_eq!(
            p1w16::grind(v, &inner, 0, 1 << 16, 64).expect("device"),
            None
        );
    }
}

/// ★ The microbenchmark (the box job reads its `BENCH` lines).
#[test]
#[ignore = "microbenchmark: a 2^29 codeword by default; run on the box with --ignored"]
fn p1w16_microbench() {
    let log_len: u32 = std::env::var("P1W16_LOG_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(29);
    let reps: u32 = std::env::var("P1W16_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let lines = p1w16::microbench(log_len, reps, 1 << 28).expect("device");
    assert!(lines.len() >= 10, "every arm printed");
}
