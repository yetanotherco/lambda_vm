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

/// Smallest per-launch block, so a tiny grinding factor still fills the grid.
const MIN_BLOCK: u64 = 1 << 18;

/// Largest per-launch block, so a huge grinding factor does not ask for an
/// absurd single launch. A miss just advances `base` and relaunches.
const MAX_BLOCK: u64 = 1 << 28;

/// ★ `LAMBDA_VM_GRIND_SCAN_FACTOR` — how many expected hit distances one launch
/// covers. Default [`SCAN_FACTOR_DEFAULT`]; read once per process and printed.
///
/// # ⛔ This is a CEILING on the block, not a multiplier on the work
///
/// Both kernels carry an early exit — `if (nonce >= *result) break;` against a
/// `volatile` result the `atomicMin` writes through L2 — and the stride walk
/// `for (i = tid; i < count; i += gridDim*blockDim)` gives every nonce in
/// `[base, base+count)` exactly one owner. So once the first valid nonce `h` is
/// recorded, every thread stops within one stride round: the permutations
/// actually executed are `h + stride`. Launches before the hitting one cover
/// exactly the part of `[0, h)` below it, so the total over the whole search is
/// `h + stride` **for any scan factor**.
///
/// What the factor buys is only the probability that one launch suffices,
/// `P = 1 − e^−k`: 99.97% at the default 8, 63% at 1. Lowering it removes no
/// permutations — they were never executed — and adds `1/(1 − e^−k)` expected
/// launches, each a sentinel H2D, a launch, an 8-byte D2H and a stream
/// synchronize.
///
/// ⇒ The knob exists so that reading is measurable on the card rather than
/// argued from the source. Nothing here touches the grinding factor itself
/// (the security parameter), the kernels, or the host re-validation of every
/// nonce the device returns.
pub const SCAN_FACTOR_ENV: &str = "LAMBDA_VM_GRIND_SCAN_FACTOR";

/// The default, and the posture every recorded measurement was taken under.
pub const SCAN_FACTOR_DEFAULT: u32 = 8;

/// What [`SCAN_FACTOR_ENV`] accepts, inclusive, and what its error names.
///
/// 0 is refused rather than clamped: `0 * expected` is 0, which the clamp would
/// turn into a fixed [`MIN_BLOCK`] block — a different search, not a smaller
/// one, and silently so.
pub const SCAN_FACTOR_RANGE: std::ops::RangeInclusive<u32> = 1..=64;

/// ★ The scan factor for this process, read once and cached.
///
/// Prints on every setting including the default, so a log that quotes the
/// factor can be shown to have read it rather than assumed it. Aborts on a
/// value outside [`SCAN_FACTOR_RANGE`] for the reason `LAMBDA_VM_WHIR_HASH`
/// aborts on an unknown hash: a measurement taken under a silently ignored
/// knob is worse than no measurement.
fn scan_factor() -> u32 {
    static FACTOR: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FACTOR.get_or_init(|| {
        let factor = match std::env::var(SCAN_FACTOR_ENV) {
            Err(_) => SCAN_FACTOR_DEFAULT,
            Ok(raw) => parse_scan_factor(raw.trim()).unwrap_or_else(|| {
                // eprintln then abort rather than a panic: a configuration
                // error at startup, where the operator needs the accepted
                // range and not a backtrace through the prover.
                eprintln!(
                    "{SCAN_FACTOR_ENV}={raw:?} is not a scan factor this path accepts. \
                     Accepted: an integer in {}..={}.",
                    SCAN_FACTOR_RANGE.start(),
                    SCAN_FACTOR_RANGE.end()
                );
                std::process::abort()
            }),
        };
        println!("★ GRIND SCAN FACTOR: {factor}");
        factor
    })
}

/// ★ The scan factor this process is searching under — the SAME cached value
/// [`search`] uses, not a second read of the environment.
///
/// Exists so a measurement can print the knob it actually ran under rather
/// than the one its launcher believed it exported (a quoted knob has to be
/// shown read on the path). Calling it also forces the banner, so a log with no
/// `★ GRIND SCAN FACTOR:` line is a run that never reached the device search.
pub fn scan_factor_in_effect() -> u32 {
    scan_factor()
}

/// ★ The nonces one launch would cover at `grinding_factor` under the scan
/// factor in effect — what [`search`] passes the kernel as `count`.
pub fn per_launch_block(grinding_factor: u8) -> u64 {
    block_size(grinding_factor, scan_factor())
}

/// The accepted spellings: a plain decimal integer inside the range.
fn parse_scan_factor(raw: &str) -> Option<u32> {
    let factor: u32 = raw.parse().ok()?;
    SCAN_FACTOR_RANGE.contains(&factor).then_some(factor)
}

/// Nonces one launch covers: the expected hit distance `2^grinding_factor`
/// times the scan factor, clamped to `[MIN_BLOCK, MAX_BLOCK]`.
///
/// `2^grinding_factor` can overflow u64 (factor 64), so saturate.
fn block_size(grinding_factor: u8, scan_factor: u32) -> u64 {
    let expected = 1u64.checked_shl(grinding_factor as u32).unwrap_or(u64::MAX);
    expected
        .saturating_mul(scan_factor as u64)
        .clamp(MIN_BLOCK, MAX_BLOCK)
}

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
///
/// ★ **The returned nonce is therefore a function of the inner hash and the
/// grinding factor alone** — the blocks are contiguous from 0 and the first one
/// to hit returns its minimum, so the block SIZE cannot move it. That is what
/// `tests/grinding.rs::gpu_grind_returns_smallest_valid_nonce` pins, and it is
/// why sweeping [`SCAN_FACTOR_ENV`] moves no proof byte.
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

    let count = block_size(grinding_factor, scan_factor());

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

#[cfg(test)]
mod tests {
    //! The knob's parser and its block arithmetic, card-free: nothing here
    //! touches `backend()`, so these run wherever the crate compiles.
    //!
    //! `scan_factor()` itself is deliberately NOT exercised — it caches in a
    //! `OnceLock`, so a test that set the variable would fix the value for
    //! every other test in the binary and the second assertion would pass on
    //! the first one's cache. The sweep sets it per PROCESS, which is what the
    //! bench arms do.

    use super::{MAX_BLOCK, MIN_BLOCK, SCAN_FACTOR_DEFAULT, block_size, parse_scan_factor};

    /// Every arm of the sweep parses, and an untouched environment is the
    /// default. Spelled as the four values the sweep actually launches, so a
    /// parser that stopped accepting one of them reddens here.
    #[test]
    fn the_parser_accepts_every_arm_of_the_sweep() {
        for k in [8u32, 4, 2, 1] {
            assert_eq!(
                parse_scan_factor(&k.to_string()),
                Some(k),
                "scan factor {k} is a sweep arm and must parse"
            );
        }
        assert_eq!(parse_scan_factor("64"), Some(64), "the top of the range");
        assert_eq!(parse_scan_factor("16"), Some(16), "inside the range");
    }

    /// The refusing half, executed. Each of these would otherwise reach the
    /// clamp and become a DIFFERENT search that no log could distinguish from
    /// the one the operator asked for.
    #[test]
    fn the_parser_refuses_what_would_silently_change_the_search() {
        for raw in ["0", "65", "abc", "", "-1", "8.0", "0x8", " ", "4294967296"] {
            assert_eq!(
                parse_scan_factor(raw),
                None,
                "{raw:?} must be refused, not defaulted"
            );
        }
    }

    /// The block size at the production grinding factor, written out as
    /// arithmetic rather than as the expression under test.
    ///
    /// ⇒ A drifted default or a dropped multiply reddens here by value.
    #[test]
    fn the_block_is_the_hit_distance_times_the_scan_factor() {
        assert_eq!(block_size(20, 8), 8 * 1_048_576, "2^20 * 8 = 2^23");
        assert_eq!(block_size(20, 4), 4 * 1_048_576, "2^20 * 4 = 2^22");
        assert_eq!(block_size(20, 2), 2 * 1_048_576, "2^20 * 2 = 2^21");
        assert_eq!(block_size(20, 1), 1_048_576, "2^20 * 1 = 2^20");
    }

    /// The default is the record posture: every measurement in the campaign
    /// was taken at a 2^23 block, and the ABBA is what may move it.
    #[test]
    fn the_default_is_the_record_posture() {
        assert_eq!(SCAN_FACTOR_DEFAULT, 8, "the record posture's scan factor");
        assert_eq!(
            block_size(20, SCAN_FACTOR_DEFAULT),
            1 << 23,
            "the record posture's per-launch block at grinding factor 20"
        );
    }

    /// Both ends of the clamp still bind, including the u64 overflow the
    /// saturating multiply exists for.
    #[test]
    fn the_clamp_binds_at_both_ends() {
        assert_eq!(
            block_size(12, 1),
            MIN_BLOCK,
            "the min-factor gate's own factor, at the narrowest scan, floors"
        );
        assert_eq!(block_size(26, 8), MAX_BLOCK, "2^26 * 8 = 2^29 ceils");
        assert_eq!(
            block_size(64, 8),
            MAX_BLOCK,
            "2^64 saturates before the multiply, then ceils"
        );
        assert!(
            (MIN_BLOCK..=MAX_BLOCK).contains(&block_size(20, 8)),
            "the record posture sits strictly inside the clamp, so neither \
             end is silently setting it"
        );
    }
}
