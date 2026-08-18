//! Parity: the GPU proof-of-work nonce search must agree with the host
//! predicate. Runs on the merge-queue GPU box via `make test-math-cuda`
//! (`cargo test -p math-cuda --release`) — `device::backend()` inside
//! `generate_nonce_gpu` requires a real GPU, like the other tests here.
//!
//! Uses real grinding factors (>= the min-factor gate). The end-to-end prover
//! suite only exercises `grinding_factor: 1`, where `limit = 1 << 63` lets a
//! broken kernel return an accepted nonce ~half the time; these factors make a
//! wrong kernel fail deterministically.

use stark::grinding::{get_inner_hash, is_valid_nonce};

fn lanes_for(seed: &[u8; 32], factor: u8) -> [u64; 4] {
    let inner = get_inner_hash(seed, factor);
    core::array::from_fn(|i| u64::from_le_bytes(inner[i * 8..i * 8 + 8].try_into().unwrap()))
}

/// At a moderate factor the kernel returns a valid nonce, and it is the
/// smallest one (the exhaustive CPU scan below it is cheap at factor 14).
#[test]
fn gpu_grind_returns_smallest_valid_nonce() {
    let seed = [14u8; 32];
    let factor = 14u8;
    let nonce = math_cuda::grinding::generate_nonce_gpu(&lanes_for(&seed, factor), factor)
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
    let nonce = math_cuda::grinding::generate_nonce_gpu(&lanes_for(&seed, factor), factor)
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
        math_cuda::grinding::generate_nonce_gpu(&lanes_for(&seed, 1), 1).is_none(),
        "GPU grind should decline factor 1"
    );
}
