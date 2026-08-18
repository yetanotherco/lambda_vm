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
