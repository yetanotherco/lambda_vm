//! Proof-of-work grinding, over whichever hash the proof's configuration
//! transcripts with.
//!
//! The construction is two hashes of one block each — 41 bytes inner, 40 bytes
//! outer — so it costs two compressions whichever hash `D` is, and the seed and
//! digest are `[u8; 32]` on both sides. Swapping the hash is therefore a type
//! substitution with no change to the shape of anything: the seed is
//! `transcript.state()`, which is 32 bytes for every transcript configuration.
//!
//! `D` is deliberately a parameter rather than a default: the PoW hash has to
//! be the proof's hash, and a defaulted one would silently keep grinding on
//! keccak for a configuration that had moved everything else.

use digest::{Digest, OutputSizeUser, typenum::U32};
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
pub fn is_valid_nonce<D>(seed: &[u8; 32], nonce: u64, grinding_factor: u8) -> bool
where
    D: Digest + OutputSizeUser<OutputSize = U32>,
{
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    let inner_hash = get_inner_hash::<D>(seed, grinding_factor);
    let limit = 1 << (64 - grinding_factor);
    is_valid_nonce_for_inner_hash::<D>(&inner_hash, nonce, limit)
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
pub fn generate_nonce<D>(seed: &[u8; 32], grinding_factor: u8) -> Option<u64>
where
    D: Digest + OutputSizeUser<OutputSize = U32>,
{
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    let inner_hash = get_inner_hash::<D>(seed, grinding_factor);
    let limit = 1 << (64 - grinding_factor);

    #[cfg(not(feature = "parallel"))]
    return (0..u64::MAX).find(|&candidate_nonce| {
        is_valid_nonce_for_inner_hash::<D>(&inner_hash, candidate_nonce, limit)
    });

    #[cfg(feature = "parallel")]
    return (0..u64::MAX).into_par_iter().find_any(|&candidate_nonce| {
        is_valid_nonce_for_inner_hash::<D>(&inner_hash, candidate_nonce, limit)
    });
}

/// Checks if the leftmost 8 bytes of `Hash(inner_hash || candidate_nonce)` are less than `limit`
/// when interpreted as `u64`.
#[inline(always)]
fn is_valid_nonce_for_inner_hash<D>(inner_hash: &[u8; 32], candidate_nonce: u64, limit: u64) -> bool
where
    D: Digest + OutputSizeUser<OutputSize = U32>,
{
    let mut data = [0; 40];
    data[..32].copy_from_slice(inner_hash);
    data[32..].copy_from_slice(&candidate_nonce.to_be_bytes());

    let digest = D::digest(data);

    let seed_head = u64::from_be_bytes(digest[..8].try_into().unwrap());
    seed_head < limit
}

/// Returns the bit-string constructed as
/// Hash(prefix || seed || grinding_factor)
/// `prefix` is the bit-string `0x123456789abcded`
fn get_inner_hash<D>(seed: &[u8; 32], grinding_factor: u8) -> [u8; 32]
where
    D: Digest + OutputSizeUser<OutputSize = U32>,
{
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&PREFIX);
    inner_data[8..40].copy_from_slice(seed);
    inner_data[40] = grinding_factor;

    let digest = D::digest(inner_data);
    digest[..32].try_into().unwrap()
}

/// The inner hash as the four little-endian u64 lanes Keccak absorbs it into —
/// the form the keccak device nonce search takes as input.
///
/// The GPU dispatch and its test both go through here rather than each doing
/// their own byte-to-lane conversion: a second copy would let this one drift
/// (`from_le_bytes` → `from_be_bytes` reads identically at a glance) with every
/// test still green, while at runtime `is_valid_nonce` rejected every device
/// nonce and the search silently sat on the CPU fallback forever.
pub fn inner_hash_lanes<D>(seed: &[u8; 32], grinding_factor: u8) -> [u64; 4]
where
    D: Digest + OutputSizeUser<OutputSize = U32>,
{
    let inner_hash = get_inner_hash::<D>(seed, grinding_factor);
    core::array::from_fn(|i| u64::from_le_bytes(inner_hash[i * 8..i * 8 + 8].try_into().unwrap()))
}

/// The inner hash as the four **big-endian** felts an algebraic digest absorbs
/// it into — the form the RPX device nonce search takes as input.
///
/// ⚠ The endianness is the whole difference from [`inner_hash_lanes`], and it
/// is not cosmetic. An algebraic digest's `felts_from_bytes` reads consecutive
/// eight-byte groups big-endian, so these four `u64`s ARE the felts the host
/// sponge absorbs; keccak reads its lanes little-endian. Crossing the two
/// compiles and runs, and produces a device search for a nonce under a message
/// the host never hashes — every returned nonce rejected, the fallback taken
/// every table, and nothing louder than one warning line to say so. That is why
/// there are two named functions and not one with a flag.
///
/// The four values are already canonical: the inner hash is an algebraic
/// digest's own output, which `digest_to_commitment` writes as four canonical
/// big-endian `u64`s. Nothing here reduces them, and the device does not either.
pub fn inner_hash_felts<D>(seed: &[u8; 32], grinding_factor: u8) -> [u64; 4]
where
    D: Digest + OutputSizeUser<OutputSize = U32>,
{
    let inner_hash = get_inner_hash::<D>(seed, grinding_factor);
    core::array::from_fn(|i| u64::from_be_bytes(inner_hash[i * 8..i * 8 + 8].try_into().unwrap()))
}

/// Grind on the GPU when a CUDA backend is up and the configuration's hash has
/// a grind kernel, falling back to the CPU search otherwise (or on any device
/// error).
///
/// ★ **`commitment_hash` is `H::COMMITMENT_HASH` at the call site, never the
/// global `config::COMMITMENT_HASH`.** The global names the *aliases'* hash and
/// nothing else: the block path proves under `prover::hash_pin::BlockStarkHash`,
/// which is pinned separately precisely so the two can differ, so a dispatch
/// keyed on the global would read BLAKE3 under the RPX pin and this whole path
/// would be dead code that still compiled.
///
/// ⚠ The keccak arm additionally checks the concrete digest by `TypeId`, the
/// way the merkle backends' keccak fast paths do. The RPX arm cannot: its
/// digest is `prover::lfm::algebraic_commit::AlgebraicDigest<RpxCommit>`, and
/// `prover` depends on this crate rather than the reverse, so there is no type
/// here to name. What closes that gap is the unconditional host validation
/// below — a configuration that said `Rpx256` and transcripted with something
/// else would lose one device search per table and fall back loudly, never
/// append an unverifiable nonce.
///
/// Which valid nonce comes back depends on the arm: the device search returns
/// the smallest in the range it scanned, while the CPU's `find_any` returns an
/// arbitrary one. Neither is a contract — the verifier accepts any nonce
/// passing `is_valid_nonce`, and nothing downstream depends on the choice. The
/// heavy per-table-per-epoch ~2^grinding_factor hashing is the prover's
/// dominant CPU cost, so this moves it off the cores onto the idle GPU.
#[cfg(feature = "cuda")]
pub fn generate_nonce_maybe_gpu<D>(
    seed: &[u8; 32],
    grinding_factor: u8,
    commitment_hash: crate::config::CommitmentHash,
) -> Option<u64>
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    use crate::config::CommitmentHash;

    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );

    // Pick the arm before doing any work. A configuration with no grind kernel
    // (BLAKE3, RPO256, Poseidon) goes straight to the host search — the
    // unconditional validation below would reject a cross-hash nonce anyway,
    // but only after wasting the full ~2^grinding_factor device search and
    // logging a spurious invalid-nonce warning every table.
    enum Arm {
        Keccak256,
        Rpx256,
    }
    let arm = match commitment_hash {
        CommitmentHash::Keccak256
            if core::any::TypeId::of::<D>()
                == core::any::TypeId::of::<crypto::hash::platform_keccak::PlatformKeccak256>() =>
        {
            Arm::Keccak256
        }
        CommitmentHash::Rpx256 => Arm::Rpx256,
        _ => return generate_nonce::<D>(seed, grinding_factor),
    };

    // Kill switch (presence-based, matching `LAMBDA_VM_NO_GPU_LOGUP`):
    // `LAMBDA_VM_NO_GPU_GRIND` forces the CPU search — a production escape hatch
    // and fallback-path coverage. Cached; read once.
    static GPU_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *GPU_DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_GRIND").is_some()) {
        return generate_nonce::<D>(seed, grinding_factor);
    }

    let found = match arm {
        Arm::Keccak256 => math_cuda::grinding::generate_nonce_gpu(
            &inner_hash_lanes::<D>(seed, grinding_factor),
            grinding_factor,
        ),
        Arm::Rpx256 => math_cuda::grinding::generate_nonce_rpx_gpu(
            &inner_hash_felts::<D>(seed, grinding_factor),
            grinding_factor,
        ),
    };

    if let Some(nonce) = found {
        // Validate unconditionally (one host hash against the ~2^grinding_factor
        // device search): a kernel/driver defect must degrade to the CPU search,
        // never append an unverifiable nonce to the transcript. This runs in
        // release too — the cost is negligible next to the grind it replaces.
        if is_valid_nonce::<D>(seed, nonce, grinding_factor) {
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
    generate_nonce::<D>(seed, grinding_factor)
}

#[cfg(not(feature = "cuda"))]
pub fn generate_nonce_maybe_gpu<D>(
    seed: &[u8; 32],
    grinding_factor: u8,
    _commitment_hash: crate::config::CommitmentHash,
) -> Option<u64>
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    generate_nonce::<D>(seed, grinding_factor)
}
