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
///
/// Public so the GPU parity test can build the same inner-hash lanes the
/// device kernel searches over.
pub fn get_inner_hash(seed: &[u8; 32], grinding_factor: u8) -> [u8; 32] {
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&PREFIX);
    inner_data[8..40].copy_from_slice(seed);
    inner_data[40] = grinding_factor;

    let digest = Keccak256::digest(inner_data);
    digest[..32].try_into().unwrap()
}

/// Grind on the GPU when a CUDA backend is up, falling back to the CPU search
/// otherwise (or on any device error). The nonce is the smallest valid one in
/// the searched range, which — like the CPU's — the verifier accepts by
/// checking `is_valid_nonce`; nothing downstream depends on which valid nonce
/// is chosen. The heavy per-table-per-epoch ~2^grinding_factor hashing is the
/// prover's dominant CPU cost, so this moves it off the 16 cores onto the idle
/// GPU.
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
    let inner_hash = get_inner_hash(seed, grinding_factor);
    // Keccak reads the 32-byte inner hash as four little-endian lanes.
    let inner_lanes: [u64; 4] = core::array::from_fn(|i| {
        u64::from_le_bytes(inner_hash[i * 8..i * 8 + 8].try_into().unwrap())
    });
    if let Some(nonce) = math_cuda::grinding::generate_nonce_gpu(&inner_lanes, grinding_factor) {
        // Validate unconditionally (one host hash against the ~2^grinding_factor
        // device search): a kernel/driver defect must degrade to the CPU search,
        // never append an unverifiable nonce to the transcript. This runs in
        // release too — the cost is negligible next to the grind it replaces.
        if is_valid_nonce(seed, nonce, grinding_factor) {
            crate::gpu_lde::GPU_GRIND_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(nonce);
        }
        log::warn!("GPU grind returned an invalid nonce ({nonce}); falling back to CPU search");
    }
    generate_nonce(seed, grinding_factor)
}

#[cfg(not(feature = "cuda"))]
pub fn generate_nonce_maybe_gpu(seed: &[u8; 32], grinding_factor: u8) -> Option<u64> {
    generate_nonce(seed, grinding_factor)
}
