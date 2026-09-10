//! The GPU dispatch for [`crypto::grinding`], whose primitive this re-exports
//! so existing call sites read unchanged.

pub use crypto::grinding::{generate_nonce, inner_hash_lanes, is_valid_nonce};

/// Grind on the GPU when a CUDA backend is up, falling back to the CPU search
/// otherwise (or on any device error). Which valid nonce comes back depends on
/// the arm: the device search returns the smallest in the range it scanned,
/// while the CPU's `find_any` returns an arbitrary one. Neither is a contract —
/// the verifier accepts any nonce passing `is_valid_nonce`, and nothing
/// downstream depends on the choice. The heavy per-table-per-epoch
/// ~2^grinding_factor hashing is the prover's dominant CPU cost, so this moves
/// it off the 16 cores onto the idle GPU.
#[cfg(feature = "cuda")]
pub fn generate_nonce_maybe_gpu(seed: &[u8; 32], grinding_factor: u8) -> Option<u64> {
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    // Kill switch (presence-based, matching `LAMBDA_VM_NO_GPU_LOGUP`):
    // `LAMBDA_VM_NO_GPU_GRIND` forces the CPU search — a production escape hatch
    // and fallback-path coverage. Cached; read once.
    static GPU_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *GPU_DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_GRIND").is_some()) {
        return generate_nonce(seed, grinding_factor);
    }
    let inner_lanes = inner_hash_lanes(seed, grinding_factor);
    if let Some(nonce) = math_cuda::grinding::generate_nonce_gpu(&inner_lanes, grinding_factor) {
        // Validate unconditionally (one host hash against the ~2^grinding_factor
        // device search): a kernel/driver defect must degrade to the CPU search,
        // never append an unverifiable nonce to the transcript. This runs in
        // release too — the cost is negligible next to the grind it replaces.
        if is_valid_nonce(seed, nonce, grinding_factor) {
            crate::gpu_lde::GPU_GRIND_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(nonce);
        }
        // eprintln, not log::warn: the CLI initialises env_logger with no
        // default filter, so a warn-level line is invisible unless RUST_LOG is
        // set — and this is the only signal that the kernel has started
        // returning garbage and the feature has silently reverted to the CPU
        // search. Matches the `[gpu]` prefix the other device-decline paths use.
        eprintln!(
            "[gpu] grind returned an invalid nonce ({nonce}); falling back to the CPU search"
        );
    }
    generate_nonce(seed, grinding_factor)
}

#[cfg(not(feature = "cuda"))]
pub fn generate_nonce_maybe_gpu(seed: &[u8; 32], grinding_factor: u8) -> Option<u64> {
    generate_nonce(seed, grinding_factor)
}
