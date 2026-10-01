//! GPU proof-of-work grinding: a parallel Keccak nonce search that mirrors the
//! host `stark::grinding::generate_nonce`, offloading the ~2^grinding_factor
//! hashes it does per table per epoch from the CPU (where they dominate the
//! prove) to the otherwise-idle GPU.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::backend;

const BLOCK_DIM: u32 = 256;
const GRID_DIM: u32 = 1024;

/// Below this grinding factor the CPU search finds a valid nonce in well under
/// a microsecond, so a device launch + shared-stream `synchronize` (which also
/// stalls whatever a rayon peer queued on that stream) is pure loss. Bounce
/// those to the CPU. The production factor is 20; only tests use tiny factors.
const GRIND_MIN_FACTOR: u8 = 12;

/// Smallest nonce whose grind head is `< limit`, or `None` when the CUDA path
/// is unavailable/errors (the caller then runs the CPU search).
///
/// `inner_lanes` are the four little-endian-read u64 lanes of the 32-byte
/// inner hash — build them with `stark::grinding::inner_hash_lanes`, which is
/// what the prover and the tests here both call. `grinding_factor` (1..=64)
/// fixes `limit = 1 << (64 - grinding_factor)` and sizes the search: the
/// expected first valid nonce is ~`2^grinding_factor`, so each launch scans a
/// contiguous block several times that, from 0 upward, and the first block that
/// hits yields the globally smallest valid nonce (the kernel `atomicMin`s it).
pub fn generate_nonce_gpu(inner_lanes: &[u64; 4], grinding_factor: u8) -> Option<u64> {
    if !(GRIND_MIN_FACTOR..=64).contains(&grinding_factor) {
        return None;
    }
    let limit: u64 = 1u64 << (64 - grinding_factor);

    let be = backend().ok()?;
    let stream = be.next_stream();
    let inner_dev = stream.clone_htod(inner_lanes.as_slice()).ok()?;

    // Per-launch block size: ~8× the expected hit distance, clamped so tiny
    // factors still launch a full grid and huge factors don't ask for an
    // absurd single block. `2^grinding_factor` can overflow u64 (factor 64), so
    // saturate.
    let expected = 1u64.checked_shl(grinding_factor as u32).unwrap_or(u64::MAX);
    let count = expected.saturating_mul(8).clamp(1 << 18, 1 << 28);

    let cfg = LaunchConfig {
        grid_dim: (GRID_DIM, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
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
                .launch_builder(&be.grind_search)
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
