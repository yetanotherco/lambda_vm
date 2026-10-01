use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

const PREFIX: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed];

/// Checks if the bit-string `Hash(Hash(prefix || seed || grinding_factor) || nonce)`
/// has at least `grinding_factor` zeros to the left.
/// `prefix` is the bit-string `0x123456789abcded`
///
/// # Parameters
///
/// * `seed`: the input seed,
/// * `nonce`: the value to be tested,
/// * `grinding_factor`: the number of leading zeros needed; must be in `1..=64`.
///
/// # Returns
///
/// `true` if the number of leading zeros is at least `grinding_factor`, and `false` otherwise.
pub fn is_valid_nonce(seed: &[u8; 32], nonce: u64, grinding_factor: u8) -> bool {
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    let inner_hash = get_inner_hash(seed, grinding_factor);
    let limit = 1 << (64 - grinding_factor);
    is_valid_nonce_for_inner_hash(&inner_hash, nonce, limit)
}

/// Performs grinding, returning a new nonce for the proof.
/// The nonce generated is such that:
/// Hash(Hash(prefix || seed || grinding_factor) || nonce) has at least `grinding_factor` zeros
/// to the left.
/// `prefix` is the bit-string `0x123456789abcded`
///
/// # Parameters
///
/// * `seed`: the input seed,
/// * `grinding_factor`: the number of leading zeros needed; must be in `1..=64`.
///
/// # Returns
///
/// A `nonce` satisfying the required condition.
pub fn generate_nonce(seed: &[u8; 32], grinding_factor: u8) -> Option<u64> {
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    let inner_hash = get_inner_hash(seed, grinding_factor);
    let limit = 1 << (64 - grinding_factor);

    #[cfg(not(feature = "parallel"))]
    return (0..u64::MAX).find(|&candidate_nonce| {
        is_valid_nonce_for_inner_hash(&inner_hash, candidate_nonce, limit)
    });

    #[cfg(feature = "parallel")]
    return (0..u64::MAX).into_par_iter().find_any(|&candidate_nonce| {
        is_valid_nonce_for_inner_hash(&inner_hash, candidate_nonce, limit)
    });
}

/// Checks if the leftmost 8 bytes of `Hash(inner_hash || candidate_nonce)` are less than `limit`
/// when interpreted as `u64`.
#[inline(always)]
fn is_valid_nonce_for_inner_hash(inner_hash: &[u8; 32], candidate_nonce: u64, limit: u64) -> bool {
    let mut data = [0; 40];
    data[..32].copy_from_slice(inner_hash);
    data[32..].copy_from_slice(&candidate_nonce.to_be_bytes());

    let digest = Keccak256::digest(data);

    let seed_head = u64::from_be_bytes(digest[..8].try_into().unwrap());
    seed_head < limit
}

/// Returns the bit-string constructed as
/// Hash(prefix || seed || grinding_factor)
/// `prefix` is the bit-string `0x123456789abcded`
fn get_inner_hash(seed: &[u8; 32], grinding_factor: u8) -> [u8; 32] {
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&PREFIX);
    inner_data[8..40].copy_from_slice(seed);
    inner_data[40] = grinding_factor;

    let digest = Keccak256::digest(inner_data);
    digest[..32].try_into().unwrap()
}

/// The inner hash as the four little-endian u64 lanes Keccak absorbs it into —
/// the form the device nonce search takes as input.
///
/// The GPU dispatch and its test both go through here rather than each doing
/// their own byte-to-lane conversion: a second copy would let this one drift
/// (`from_le_bytes` → `from_be_bytes` reads identically at a glance) with every
/// test still green, while at runtime `is_valid_nonce` rejected every device
/// nonce and the search silently sat on the CPU fallback forever.
pub fn inner_hash_lanes(seed: &[u8; 32], grinding_factor: u8) -> [u64; 4] {
    let inner_hash = get_inner_hash(seed, grinding_factor);
    core::array::from_fn(|i| u64::from_le_bytes(inner_hash[i * 8..i * 8 + 8].try_into().unwrap()))
}

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
