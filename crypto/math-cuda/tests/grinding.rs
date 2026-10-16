//! The GPU nonce search must produce nonces the host predicate accepts. There
//! is nothing to compare against the CPU search itself — any nonce satisfying
//! `is_valid_nonce` is as good as any other, and the CPU's `find_any` does not
//! even agree with itself between runs — so what is pinned here is validity,
//! plus the search completeness that minimality stands in for.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda`
//! (`cargo test -p math-cuda --release`) — `device::backend()` inside
//! `generate_nonce_gpu` requires a real GPU, like the other tests here.
//!
//! Uses real grinding factors (>= the min-factor gate). The end-to-end prover
//! suite only exercises `grinding_factor: 1`, where `limit = 1 << 63` lets a
//! broken kernel return an accepted nonce ~half the time; these factors make a
//! wrong kernel fail deterministically.
//!
//! The lanes come from `stark::grinding::inner_hash_lanes`, the same call the
//! prover makes — building them here instead would leave the production
//! conversion untested.

use stark::grinding::{inner_hash_lanes, is_valid_nonce};

/// At a moderate factor the kernel returns a valid nonce, and it is the
/// smallest one (the exhaustive CPU scan below it is cheap at factor 14).
///
/// Minimality is not a contract — any valid nonce would do — but it is a cheap
/// probe of search completeness: a stride or bounds bug that skipped part of
/// the range would still return a *valid* nonce, just not the first one, and
/// plain validity checking would miss that. Deterministic despite the grid
/// being parallel, because `atomicMin` is an order-independent reduction. If a
/// future kernel drops minimality deliberately, relax this to validity rather
/// than treating the red as a defect.
#[test]
fn gpu_grind_returns_smallest_valid_nonce() {
    let seed = [14u8; 32];
    let factor = 14u8;
    let nonce = math_cuda::grinding::generate_nonce_gpu(&inner_hash_lanes(&seed, factor), factor)
        .expect("GPU grind (needs a GPU)");
    assert!(
        is_valid_nonce(&seed, nonce, factor),
        "GPU nonce {nonce} fails is_valid_nonce (factor {factor})"
    );
    assert!(
        (0..nonce).all(|n| !is_valid_nonce(&seed, n, factor)),
        "GPU nonce {nonce} is not the smallest valid nonce (factor {factor})"
    );
}

/// At the production factor the kernel returns a valid nonce (validity only —
/// scanning 0..nonce would be ~2^20 hashes).
#[test]
fn gpu_grind_valid_at_production_factor() {
    let seed = [20u8; 32];
    let factor = 20u8;
    let nonce = math_cuda::grinding::generate_nonce_gpu(&inner_hash_lanes(&seed, factor), factor)
        .expect("GPU grind (needs a GPU)");
    assert!(
        is_valid_nonce(&seed, nonce, factor),
        "GPU nonce {nonce} fails is_valid_nonce (factor {factor})"
    );
}

/// Below the min-factor gate the GPU path declines (→ CPU search), so the tiny
/// factors every non-GPU-benchmark test uses never pay a launch.
#[test]
fn gpu_grind_declines_below_min_factor() {
    let seed = [1u8; 32];
    assert!(
        math_cuda::grinding::generate_nonce_gpu(&inner_hash_lanes(&seed, 1), 1).is_none(),
        "GPU grind should decline factor 1"
    );
}
