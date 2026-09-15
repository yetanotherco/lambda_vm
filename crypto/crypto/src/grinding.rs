//! Proof of work over a transcript state: the prover finds a nonce whose hash
//! has `grinding_factor` leading zeros, and absorbs it before the next
//! challenge is drawn.
//!
//! Retrying that challenge then costs `2^grinding_factor` hashes, which is what
//! lets a query count buy more soundness than its own bits.
//!
//! Lives here rather than in a proof-system crate because both the univariate
//! prover and the multilinear one grind against the same primitive, and so does
//! the device dispatch below: `multilinear` cannot reach `stark`, which depends
//! on it.
//!
//! # The hash is a parameter, with NO default
//!
//! The construction is two hashes of one block each — 41 bytes inner, 40 bytes
//! outer — so it costs two compressions whichever hash `D` is, and the seed and
//! the digest are `[u8; 32]` on both sides. Swapping the hash is therefore a
//! type substitution that changes the shape of nothing.
//!
//! `D` is deliberately a parameter rather than a defaulted one: the
//! proof-of-work hash has to be the proof's hash, and a defaulted `D` would
//! silently keep grinding on keccak for a configuration that had moved
//! everything else — self-consistent between prover and verifier, and therefore
//! silent. Every call site states its hash.

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
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    let inner_hash = get_inner_hash::<D>(seed, grinding_factor);
    let limit = 1 << (64 - grinding_factor);
    is_valid_nonce_for_inner_hash::<D>(&inner_hash, nonce, limit)
}

/// ⚠ **A ground proof is NOT byte-reproducible, and the reason is worse than
/// it looks.**
///
/// [`generate_nonce`] returns *a* valid nonce, not *the* valid nonce: under
/// `parallel` it is rayon's `find_any`, which hands back whichever worker
/// finished first, and on the device arm it is whatever the kernel's scan
/// reached. That much is a known property. What makes it load-bearing for any
/// byte gate is what happens next: **the nonce is absorbed into the
/// transcript** (`multilinear::whir_chain::grind`, and the univariate prover
/// likewise), so every challenge drawn after the first grind depends on which
/// valid nonce the search happened to return. Two honest runs therefore differ
/// in every Merkle root, every out-of-domain value and every opening from the
/// first grind onward — and **no post-hoc normalisation of the nonce fields can
/// recover the agreement**, because the divergence is not in the nonce fields.
///
/// [`deterministic`] is the escape hatch a byte gate needs: with
/// `LAMBDA_VM_DETERMINISTIC_GRIND` set, the search returns the SMALLEST valid
/// nonce, which is a function of the seed alone, so the whole proof becomes
/// reproducible. It is off by default and changes nothing about validity — the
/// verifier accepts any nonce passing [`is_valid_nonce`] — only about which of
/// them is chosen. Prove time rises: the smallest-first search cannot stop at
/// the first hit any worker finds.
///
/// Read once, cached, presence-based — the convention `LAMBDA_VM_NO_GPU_GRIND`
/// uses.
#[cfg(feature = "std")]
pub fn deterministic() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some())
}

#[cfg(not(feature = "std"))]
pub fn deterministic() -> bool {
    false
}

/// Performs grinding, returning a new nonce for the proof.
/// The nonce generated is such that:
/// Hash(Hash(prefix || seed || grinding_factor) || nonce) has at least `grinding_factor` zeros
/// to the left.
/// `prefix` is the bit-string `0x123456789abcded`
///
/// Which valid nonce comes back is NOT a contract — see [`deterministic`] for
/// the one case where it is, and for why that matters to a byte gate.
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
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    if deterministic() {
        return generate_nonce_smallest::<D>(seed, grinding_factor);
    }
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

/// ★ The SMALLEST valid nonce — a function of the seed and the factor alone,
/// and therefore the thing a byte gate can reproduce.
///
/// Exposed as its own entry point rather than reachable only through the
/// environment, so the property "this is reproducible, and it is genuinely the
/// smallest" is testable without a process-global switch. `find_first` rather
/// than `find_any`: rayon prunes candidates above the best hit so far, so the
/// cost is bounded by the smallest hit's index rather than by the whole range,
/// but it cannot stop as early as `find_any` and that is the price of the
/// property.
pub fn generate_nonce_smallest<D>(seed: &[u8; 32], grinding_factor: u8) -> Option<u64>
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
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
    return (0..u64::MAX)
        .into_par_iter()
        .find_first(|&candidate_nonce| {
            is_valid_nonce_for_inner_hash::<D>(&inner_hash, candidate_nonce, limit)
        });
}

/// Successful GPU grind dispatches — one per nonce search that ran on device
/// and produced a nonce the host check accepted (a device miss or an invalid
/// kernel result falls back to the CPU search and is not counted).
#[cfg(feature = "cuda")]
static GPU_GRIND_CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "cuda")]
pub fn gpu_grind_calls() -> u64 {
    GPU_GRIND_CALLS.load(core::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
pub fn reset_gpu_grind_calls() {
    GPU_GRIND_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Grind on the GPU when a CUDA backend is up, falling back to the CPU search
/// otherwise (or on any device error). Which valid nonce comes back depends on
/// the arm: the device search returns the smallest in the range it scanned,
/// while the CPU's `find_any` returns an arbitrary one. Neither is a contract —
/// the verifier accepts any nonce passing [`is_valid_nonce`], and nothing
/// downstream depends on the choice.
///
/// ★ The device search is keccak's, so the dispatch is guarded on `D` being the
/// platform keccak digest — the same `TypeId` discipline the Merkle backends'
/// guest fast paths use. A configuration whose hash has no device kernel takes
/// the CPU search and is CORRECT there, rather than being handed a nonce some
/// other hash's kernel found: `is_valid_nonce::<D>` would reject such a nonce
/// anyway, but only after the search had already cost the run, and a silent
/// permanent fallback is exactly the failure `inner_hash_lanes`'s own note
/// describes.
#[cfg(feature = "cuda")]
pub fn generate_nonce_maybe_gpu<D>(seed: &[u8; 32], grinding_factor: u8) -> Option<u64>
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    debug_assert!(
        (1..=64).contains(&grinding_factor),
        "grinding_factor must be in 1..=64, got {grinding_factor}"
    );
    // Kill switch (presence-based, matching `LAMBDA_VM_NO_GPU_LOGUP`):
    // `LAMBDA_VM_NO_GPU_GRIND` forces the CPU search — a production escape hatch
    // and fallback-path coverage. Cached; read once.
    static GPU_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *GPU_DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_GRIND").is_some()) {
        return generate_nonce::<D>(seed, grinding_factor);
    }
    // The device search returns the smallest nonce in the range IT scanned,
    // which is not the same promise as the smallest that exists. Under the
    // deterministic knob the host search is the only one that makes the
    // promise, so the device steps aside rather than being trusted to keep it.
    if deterministic() {
        return generate_nonce::<D>(seed, grinding_factor);
    }
    if core::any::TypeId::of::<D>()
        != core::any::TypeId::of::<crate::hash::platform_keccak::PlatformKeccak256>()
    {
        return generate_nonce::<D>(seed, grinding_factor);
    }
    let inner_lanes = inner_hash_lanes::<D>(seed, grinding_factor);
    if let Some(nonce) = math_cuda::grinding::generate_nonce_gpu(&inner_lanes, grinding_factor) {
        // Validate unconditionally (one host hash against the ~2^grinding_factor
        // device search): a kernel/driver defect must degrade to the CPU search,
        // never append an unverifiable nonce to the transcript.
        if is_valid_nonce::<D>(seed, nonce, grinding_factor) {
            GPU_GRIND_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            return Some(nonce);
        }
        // eprintln, not log::warn: the CLI initialises env_logger with no
        // default filter, so a warn-level line is invisible unless RUST_LOG is
        // set — and this is the only signal that the kernel has started
        // returning garbage and the feature has silently reverted to the CPU
        // search.
        eprintln!(
            "[gpu] grind returned an invalid nonce ({nonce}); falling back to the CPU search"
        );
    }
    generate_nonce::<D>(seed, grinding_factor)
}

#[cfg(not(feature = "cuda"))]
pub fn generate_nonce_maybe_gpu<D>(seed: &[u8; 32], grinding_factor: u8) -> Option<u64>
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    generate_nonce::<D>(seed, grinding_factor)
}

/// Checks if the leftmost 8 bytes of `Hash(inner_hash || candidate_nonce)` are less than `limit`
/// when interpreted as `u64`.
#[inline(always)]
fn is_valid_nonce_for_inner_hash<D>(inner_hash: &[u8; 32], candidate_nonce: u64, limit: u64) -> bool
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    // Tag this finalize as grinding so a verify-hash metric can report it apart
    // (see `crate::hash_metrics`); no-op unless the `hash-metrics` feature is on.
    // The metric is keccak-only by construction, so a non-keccak `D` leaves it
    // at zero rather than reporting another hash's work as keccak's.
    crate::hash_metrics::count_grinding();
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
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    // Grinding finalize (see `crate::hash_metrics`); no-op unless enabled.
    crate::hash_metrics::count_grinding();
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&PREFIX);
    inner_data[8..40].copy_from_slice(seed);
    inner_data[40] = grinding_factor;

    let digest = D::digest(inner_data);
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
pub fn inner_hash_lanes<D>(seed: &[u8; 32], grinding_factor: u8) -> [u64; 4]
where
    D: Digest + OutputSizeUser<OutputSize = U32> + 'static,
{
    let inner_hash = get_inner_hash::<D>(seed, grinding_factor);
    core::array::from_fn(|i| u64::from_le_bytes(inner_hash[i * 8..i * 8 + 8].try_into().unwrap()))
}
