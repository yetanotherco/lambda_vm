//! GPU proof-of-work grinding: a parallel nonce search that mirrors the host
//! `crypto::grinding::generate_nonce`, offloading the ~2^grinding_factor hashes
//! it does per grind from the CPU — where they dominate the prove — to the
//! otherwise-idle GPU.
//!
//! Two arms, one per outer hash: [`generate_nonce_gpu`] for keccak-256 and
//! [`generate_nonce_rpx_gpu`] for RPX256. They differ in the kernel and in **how
//! the 32-byte inner hash is read into four `u64`s** — LITTLE-endian lanes for
//! keccak, BIG-endian felts for RPX. Everything else (the min-factor gate, the
//! block sizing, the sentinel loop, the first-hit reduction) is one policy,
//! written once in [`search`].
//!
//! # ⚠ Why the two entry points are named rather than flagged
//!
//! The endianness is the whole difference and it is not cosmetic. An algebraic
//! digest reads consecutive eight-byte groups big-endian, so those four `u64`s
//! ARE the felts the host sponge absorbs; keccak reads its lanes little-endian.
//! Crossing them compiles, runs, and searches for a nonce under a message the
//! host never hashes: every returned nonce fails the host check, the prover
//! falls back to the CPU on every grind, and nothing says so louder than one
//! warning line. Two functions with two doc comments is the cheapest way to
//! make that mistake hard to type.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::backend;

const BLOCK_DIM: u32 = 256;
const GRID_DIM: u32 = 1024;

/// Threads per block for the RPX arm.
///
/// Half keccak's, for the reason every RPX kernel in this crate launches
/// narrow: a thread carries a twelve-lane `u64` state plus the inverse S-box's
/// live temporaries across a non-inlined `permute` call, so occupancy is bought
/// with registers rather than threads.
const RPX_BLOCK_DIM: u32 = 128;

/// Below this grinding factor the CPU search finds a valid nonce in well under
/// a microsecond, so a device launch + shared-stream `synchronize` (which also
/// stalls whatever a rayon peer queued on that stream) is pure loss. Bounce
/// those to the CPU. The production factor is 20; only tests use tiny factors.
pub const GRIND_MIN_FACTOR: u8 = 12;

/// Which outer hash the search runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Keccak256,
    Rpx256,
}

/// Smallest nonce whose keccak grind head is `< limit`, or `None` when the CUDA
/// path is unavailable/errors (the caller then runs the CPU search).
///
/// `inner_lanes` are the four **little-endian**-read `u64` lanes of the 32-byte
/// inner hash — build them with `crypto::grinding::inner_hash_lanes`.
pub fn generate_nonce_gpu(inner_lanes: &[u64; 4], grinding_factor: u8) -> Option<u64> {
    search(Arm::Keccak256, inner_lanes, grinding_factor)
}

/// Smallest nonce whose RPX grind head is `< limit`, or `None` when the CUDA
/// path is unavailable/errors (the caller then runs the CPU search).
///
/// ⚠ `inner_felts` are the four **big-endian**-read `u64`s of the 32-byte inner
/// hash — build them with `crypto::grinding::inner_hash_felts`, never with
/// `inner_hash_lanes`. See the module header for what crossing them does.
///
/// # The preimage, stated where the kernel is called
///
/// The host predicate hashes `inner_hash ‖ nonce.to_be_bytes()` — **40 bytes,
/// which is five felts, which is one rate-8 block and therefore exactly one
/// permutation**. Its capacity is `leaf_capacity(5)`: lane 0 the padding flag
/// `5 mod 8 = 5`, lane 1 the LEAF domain. `rpx_grind_search` builds the same
/// block — `init(5)`, absorb `f0..f3` then `canonical(nonce)` — and compares
/// `digest[0] < limit`, which is the same number the host compares because
/// `u64::from_be_bytes(digest[..8])` IS felt 0's canonical value.
///
/// The four `inner_felts` need no reduction on either side: they are an
/// algebraic digest's own output, which `digest_to_commitment` writes as four
/// canonical big-endian `u64`s.
pub fn generate_nonce_rpx_gpu(inner_felts: &[u64; 4], grinding_factor: u8) -> Option<u64> {
    search(Arm::Rpx256, inner_felts, grinding_factor)
}

/// The range walk both arms share.
///
/// `grinding_factor` (1..=64) fixes `limit = 1 << (64 - grinding_factor)` and
/// sizes the search: the expected first valid nonce is ~`2^grinding_factor`, so
/// each launch scans a contiguous block several times that, from 0 upward, and
/// the first block that hits yields the globally smallest valid nonce (the
/// kernels `atomicMin` it).
fn search(arm: Arm, inner: &[u64; 4], grinding_factor: u8) -> Option<u64> {
    if !(GRIND_MIN_FACTOR..=64).contains(&grinding_factor) {
        return None;
    }
    let limit: u64 = 1u64 << (64 - grinding_factor);

    let be = backend().ok()?;
    let (kernel, block_dim) = match arm {
        Arm::Keccak256 => (&be.grind_search, BLOCK_DIM),
        Arm::Rpx256 => (&be.rpx_grind_search, RPX_BLOCK_DIM),
    };
    let stream = be.next_stream();
    let inner_dev = stream.clone_htod(inner.as_slice()).ok()?;

    // Per-launch block size: ~8× the expected hit distance, clamped so tiny
    // factors still launch a full grid and huge factors don't ask for an
    // absurd single block. `2^grinding_factor` can overflow u64 (factor 64), so
    // saturate.
    let expected = 1u64.checked_shl(grinding_factor as u32).unwrap_or(u64::MAX);
    let count = expected.saturating_mul(8).clamp(1 << 18, 1 << 28);

    let cfg = LaunchConfig {
        grid_dim: (GRID_DIM, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    // One reusable device slot for the running minimum, reset to the sentinel
    // (U64_MAX) before each block rather than reallocated every iteration.
    // `sentinel` is a named binding so it outlives every async H2D below.
    let sentinel = [u64::MAX];
    let mut result_dev = stream.clone_htod(&sentinel).ok()?;

    let mut base: u64 = 0;
    loop {
        stream.memcpy_htod(&sentinel, &mut result_dev).ok()?;
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&inner_dev)
                .arg(&limit)
                .arg(&base)
                .arg(&count)
                .arg(&mut result_dev)
                .launch(cfg)
                .ok()?;
        }
        let host = stream.clone_dtoh(&result_dev).ok()?;
        stream.synchronize().ok()?;
        if host[0] != u64::MAX {
            return Some(host[0]);
        }
        // Nothing in `[base, base+count)` — advance. Bail (→ CPU fallback) if
        // the block would run past u64, matching the host search's finite range.
        base = base.checked_add(count)?;
    }
}
