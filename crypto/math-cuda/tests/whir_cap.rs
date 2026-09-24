//! W1 on the device: `DeviceCodeword::paths_and_cap` returns the paths and the
//! Merkle cap of ONE rebuilt tree, and both equal the host tree's.
//!
//! Needs a GPU (`make test-math-cuda`). The reference is `multilinear`'s host
//! commitment over the same codeword (`CodewordCommitment::new` on a base
//! codeword hashes on the host): its full paths (`open_many`) and its
//! owner-encoded capped paths (`open_many_capped`), whose first path ends with
//! the host tree's cap.
//!
//! Three regimes for the leaf layer, because the cap must come from the tree
//! that was built whatever built its leaves: SERVED from the retained layer
//! (same blocking as the commit), REHASHED (another blocking, so the retained
//! layer's key does not match), and EVICTED (the layer reclaimed by the
//! allocator's evictor before the opening). In each, the cap costs no extra
//! tree build: `tree_builds` rises by exactly one per call.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use math_cuda::DeviceHash;
use multilinear::mle::Mle;
use multilinear::whir::{self, Domain};
use multilinear::whir_commit::CodewordCommitment;
use multilinear::whir_hash::{DeviceHashKey, KeccakWhir, RpxWhir, WhirHash};
use std::sync::Mutex;

type FE = FieldElement<F>;

/// Every test here commits, and eviction moves the process-wide reservation
/// total, so they take turns (the `whir_tree_cache.rs` pattern).
static DEVICE_GLOBALS: Mutex<()> = Mutex::new(());

fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    DEVICE_GLOBALS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Mirror of `DeviceHashKey::into_math_cuda` (cuda-gated on `multilinear`).
fn key<H: WhirHash>() -> DeviceHash {
    match H::DEVICE {
        DeviceHashKey::Keccak256 => DeviceHash::Keccak256,
        DeviceHashKey::Rpx256 => DeviceHash::Rpx256,
    }
}

fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    let evals: Vec<FE> = (0..(1u64 << num_vars))
        .map(|i| FE::from(i.wrapping_mul(6364136223846793005).wrapping_add(seed) >> 11))
        .collect();
    Mle::new(evals).expect("power of two")
}

fn nodes(bytes: &[u8]) -> Vec<[u8; 32]> {
    bytes
        .chunks_exact(32)
        .map(|n| n.try_into().expect("32 bytes"))
        .collect()
}

/// The device result against the host tree at blocking `k`, every cap height
/// up to `min(depth, 6)`.
fn assert_matches_host<H: WhirHash>(
    name: &str,
    device: &math_cuda::whir::DeviceCodeword,
    host: &CodewordCommitment<F, H>,
    k: usize,
    positions: &[usize],
    expect_leaf_pass: impl Fn(u64) -> bool,
) {
    let depth = host.depth();
    let full = host.open_many(positions).expect("host paths");
    let pos32: Vec<u32> = positions.iter().map(|p| *p as u32).collect();
    for c in 0..=depth.min(6) {
        let builds = device.tree_builds();
        let passes = device.leaf_passes();
        let (paths, cap) = device
            .paths_and_cap(k, &pos32, c, key::<H>())
            .unwrap_or_else(|e| panic!("{name} k={k} c={c}: paths_and_cap: {e:?}"));
        assert_eq!(
            device.tree_builds(),
            builds + 1,
            "{name} k={k} c={c}: paths and cap must come from ONE tree build"
        );
        assert!(
            expect_leaf_pass(device.leaf_passes() - passes),
            "{name} k={k} c={c}: unexpected leaf-pass count {} -> {}",
            passes,
            device.leaf_passes()
        );
        let paths = nodes(&paths);
        let cap = nodes(&cap);
        assert_eq!(cap.len(), 1 << c, "{name} k={k} c={c}: cap length");
        if c == 0 {
            assert_eq!(
                cap[0],
                host.root(),
                "{name} k={k}: the height-0 cap is the root"
            );
        }
        // Full paths: byte-identical to the host tree's, query by query.
        for (q, opening) in full.iter().enumerate() {
            assert_eq!(
                &paths[q * depth..(q + 1) * depth],
                opening.proof.merkle_path.as_slice(),
                "{name} k={k} c={c}: path {q}"
            );
        }
        // The owner encoding the host produces ends with the host tree's cap.
        let capped = host
            .open_many_capped(positions, c, true)
            .expect("host capped paths");
        if c > 0 {
            assert_eq!(
                &capped[0].proof.merkle_path[depth - c..],
                cap.as_slice(),
                "{name} k={k} c={c}: the device cap is the host tree's cap"
            );
        }
    }
}

fn setup<H: WhirHash>(
    num_vars: usize,
    k_commit: usize,
) -> (math_cuda::whir::DeviceCodeword, Vec<FE>, Domain<F>) {
    let f = poly(num_vars, 3);
    let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
    let (device, _root) = math_cuda::whir::commit_codeword(&raw, 2, k_commit, false, key::<H>())
        .expect("device commit (needs a GPU)");
    let domain = Domain::<F>::new(num_vars + 2).expect("domain");
    let host_codeword =
        whir::encode::<F, F>(&whir::lift_coefficients(&f), &domain).expect("encode");
    (device, host_codeword, domain)
}

/// SERVED and REHASHED, k = 1..5, both hashes.
#[test]
fn paths_and_cap_are_the_host_trees_served_or_rehashed() {
    let _exclusive = exclusive();
    fn run<H: WhirHash>(name: &str) {
        let num_vars = 12;
        for k_commit in 1..=5usize {
            let (device, host_codeword, _) = setup::<H>(num_vars, k_commit);
            let leaves = (host_codeword.len()) >> k_commit;
            let positions = [0usize, 1, leaves / 3, leaves - 1];
            let host =
                CodewordCommitment::<_, H>::new(&host_codeword, k_commit).expect("host commit");
            // Same blocking as the commit: the retained layer is served.
            assert_matches_host(name, &device, &host, k_commit, &positions, |d| d == 0);
            // Another blocking: the layer does not match, the leaves are hashed.
            let k_other = if k_commit == 5 { 3 } else { k_commit + 1 };
            let other =
                CodewordCommitment::<_, H>::new(&host_codeword, k_other).expect("host commit");
            let leaves = host_codeword.len() >> k_other;
            let positions = [0usize, leaves / 2, leaves - 1];
            assert_matches_host(name, &device, &other, k_other, &positions, |d| d == 1);
        }
    }
    run::<KeccakWhir>("keccak");
    run::<RpxWhir>("rpx");
}

/// EVICTED: the retained layer is reclaimed by the allocator's evictor, and
/// the next opening rebuilds the whole tree — its cap still the host's.
#[test]
fn paths_and_cap_after_the_retained_layer_is_evicted() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("eviction test needs a GPU");
    let k = 4;
    let (device, host_codeword, _) = setup::<RpxWhir>(14, k);
    let layer_bytes = device.retained_leaf_bytes();
    assert!(layer_bytes > 0, "precondition: the commit retained a layer");

    let gap = layer_bytes / 2;
    let hog_bytes = be
        .vram_budget_bytes()
        .saturating_sub(be.reserved_bytes())
        .saturating_sub(gap);
    let hog = math_cuda::device::reserve(hog_bytes).expect("the hog reservation cannot fail");
    let got = math_cuda::device::reserve(layer_bytes)
        .expect("the reserve must succeed by evicting the retained layer");
    assert_eq!(device.retained_leaf_bytes(), 0, "the layer was evicted");
    drop(got);
    drop(hog);

    let host = CodewordCommitment::<_, RpxWhir>::new(&host_codeword, k).expect("host commit");
    let leaves = host_codeword.len() >> k;
    let positions = [0usize, 5, leaves / 2, leaves - 1];
    assert_matches_host("rpx evicted", &device, &host, k, &positions, |d| d >= 1);
}
