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

/// Threads per block for the keccak arm.
///
/// ⛔ NOT a knob, and neither is its RPX twin. The two block dims are tuned per
/// kernel against register pressure, which is a property of the kernel body;
/// the GRID is what a launch chooses, and that is [`GRID_ENV`].
pub const BLOCK_DIM: u32 = 256;

/// Threads per block for the RPX arm.
///
/// Half keccak's, for the reason every RPX kernel in this crate launches
/// narrow: a thread carries a twelve-lane `u64` state plus the inverse S-box's
/// live temporaries across a non-inlined `permute` call, so occupancy is bought
/// with registers rather than threads.
pub const RPX_BLOCK_DIM: u32 = 128;

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
/// covers. Default [`SCAN_FACTOR_DEFAULT`].
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

/// ★ `LAMBDA_VM_GRIND_GRID` — blocks per launch. Default [`GRID_DEFAULT`].
///
/// # Why the GRID and not the block dims
///
/// `stride = grid × block_dim` is the number of nonces the walk advances per
/// iteration, and it is the term the early exit above leaves behind: the search
/// executes `h + stride` permutations, so the overshoot past the first hit is
/// `stride/h` — 12.5% at the default against `h = 2^20`.
///
/// That gives the knob two opposite edges and the sweep has to run BOTH ways:
/// - while the card is **not** filled, a wider grid raises throughput faster
///   than it raises the overshoot, and the wall falls;
/// - once the card **is** filled, extra blocks only queue, and the wider stride
///   is pure added work — narrower is then strictly better.
///
/// Which edge the default sits on is a residency question, and residency is
/// decided by registers per thread, which is why [`BLOCK_DIM`] and
/// [`RPX_BLOCK_DIM`] are NOT knobs: they are tuned per kernel body (lane K's
/// territory), and moving them changes what an occupancy reading means.
/// [`device_fill`] reads the answer off the driver instead of estimating it.
pub const GRID_ENV: &str = "LAMBDA_VM_GRIND_GRID";

/// The default, and the posture every recorded measurement was taken under.
pub const GRID_DEFAULT: u32 = 1024;

/// What [`GRID_ENV`] accepts, inclusive.
///
/// The top is where the knob stops meaning anything rather than where CUDA
/// stops accepting it (the driver allows 2^31−1 blocks in x): at 65,536 blocks
/// the RPX stride is 8.4 M against an expected hit at 2^20, so the search would
/// be overshoot and nothing else. 0 is refused because a launch of no blocks
/// scans nothing and the retry loop would spin forever.
pub const GRID_RANGE: std::ops::RangeInclusive<u32> = 1..=65_536;

/// ★ The two launch knobs, read together so one line can print both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Knobs {
    /// How many expected hit distances one launch covers.
    pub scan: u32,
    /// Blocks per launch.
    pub grid: u32,
}

impl Knobs {
    /// The record posture: what an unset environment reads.
    pub const DEFAULT: Self = Self {
        scan: SCAN_FACTOR_DEFAULT,
        grid: GRID_DEFAULT,
    };

    /// Nonces the walk advances per iteration — `grid × block_dim`, and the
    /// term the early exit leaves behind as overshoot.
    pub const fn stride(self, block_dim: u32) -> u64 {
        self.grid as u64 * block_dim as u64
    }

    /// Nonces one launch covers: the expected hit distance `2^grinding_factor`
    /// times the scan factor, clamped to `[MIN_BLOCK, MAX_BLOCK]`.
    ///
    /// `2^grinding_factor` can overflow u64 (factor 64), so saturate.
    pub fn block(self, grinding_factor: u8) -> u64 {
        let expected = 1u64.checked_shl(grinding_factor as u32).unwrap_or(u64::MAX);
        expected
            .saturating_mul(self.scan as u64)
            .clamp(MIN_BLOCK, MAX_BLOCK)
    }
}

/// ★ The knobs for this process, read once and cached.
///
/// Prints ONE line on every setting including the defaults, carrying both
/// knobs AND the stride each arm gets — the stride is the mechanism, so it is
/// printed as a read rather than left for the reader to multiply. A log with no
/// `★ GRIND KNOBS:` line is a run that never reached the device search.
///
/// Aborts on a value outside its range for the reason `LAMBDA_VM_WHIR_HASH`
/// aborts on an unknown hash: a measurement taken under a silently ignored knob
/// is worse than no measurement.
pub fn knobs_in_effect() -> Knobs {
    static KNOBS: std::sync::OnceLock<Knobs> = std::sync::OnceLock::new();
    *KNOBS.get_or_init(|| {
        let knobs = Knobs {
            scan: read_knob(SCAN_FACTOR_ENV, SCAN_FACTOR_DEFAULT, &SCAN_FACTOR_RANGE),
            grid: read_knob(GRID_ENV, GRID_DEFAULT, &GRID_RANGE),
        };
        println!(
            "★ GRIND KNOBS: scan {} · grid {} · stride rpx {} / keccak {}",
            knobs.scan,
            knobs.grid,
            knobs.stride(RPX_BLOCK_DIM),
            knobs.stride(BLOCK_DIM),
        );
        knobs
    })
}

/// ★ The nonces one launch would cover at `grinding_factor` under the knobs in
/// effect — what [`search`] passes the kernel as `count`.
pub fn per_launch_block(grinding_factor: u8) -> u64 {
    knobs_in_effect().block(grinding_factor)
}

/// One knob, parsed from the environment or defaulted, refused outside its
/// range with the offending value and the range both named.
fn read_knob(env: &str, default: u32, range: &std::ops::RangeInclusive<u32>) -> u32 {
    match std::env::var(env) {
        Err(_) => default,
        Ok(raw) => parse_knob(raw.trim(), range).unwrap_or_else(|| {
            // eprintln then abort rather than a panic: a configuration error at
            // startup, where the operator needs the accepted range and not a
            // backtrace through the prover.
            eprintln!(
                "{env}={raw:?} is not a value this path accepts. Accepted: an integer in {}..={}.",
                range.start(),
                range.end()
            );
            std::process::abort()
        }),
    }
}

/// The accepted spellings: a plain decimal integer inside `range`.
fn parse_knob(raw: &str, range: &std::ops::RangeInclusive<u32>) -> Option<u32> {
    let value: u32 = raw.parse().ok()?;
    range.contains(&value).then_some(value)
}

/// ★ What the driver says about filling this card with the RPX grind kernel.
///
/// Every field is READ, not estimated: the residency question the grid knob
/// turns on is decided by registers per thread, and guessing that is how a
/// sweep gets sized against a card nobody measured.
#[derive(Clone, Copy, Debug)]
pub struct DeviceFill {
    /// Multiprocessors on the device.
    pub sm_count: u32,
    /// The device's own ceiling on resident threads per multiprocessor.
    pub max_threads_per_sm: u32,
    /// Registers the RPX grind kernel uses per thread.
    pub rpx_regs_per_thread: i32,
    /// Blocks of [`RPX_BLOCK_DIM`] the driver will keep resident per
    /// multiprocessor — the occupancy the register count actually buys.
    pub rpx_blocks_per_sm: u32,
    /// The block dim those blocks carry.
    pub rpx_block_dim: u32,
}

impl DeviceFill {
    /// ⭐ Blocks that can be resident at once. A grid ABOVE this queues: the
    /// extra blocks buy no parallelism and their stride is pure overshoot.
    pub const fn resident_blocks(&self) -> u64 {
        self.sm_count as u64 * self.rpx_blocks_per_sm as u64
    }

    /// Threads that can be resident at once, by the same reading.
    pub const fn resident_threads(&self) -> u64 {
        self.resident_blocks() * self.rpx_block_dim as u64
    }

    /// What fraction of the resident ceiling a grid of `grid` blocks asks for.
    /// Above 1.0 the surplus queues.
    pub fn fill(&self, grid: u32) -> f64 {
        grid as f64 / self.resident_blocks() as f64
    }
}

/// Reads [`DeviceFill`] off the driver, or `None` where there is no device.
pub fn device_fill() -> Option<DeviceFill> {
    use cudarc::driver::sys::CUdevice_attribute;
    let be = backend().ok()?;
    let sm_count = be
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()?;
    let max_threads_per_sm = be
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .ok()?;
    let rpx_regs_per_thread = be.rpx_grind_search.num_regs().ok()?;
    let rpx_blocks_per_sm = be
        .rpx_grind_search
        .occupancy_max_active_blocks_per_multiprocessor(RPX_BLOCK_DIM, 0, None)
        .ok()?;
    Some(DeviceFill {
        sm_count: sm_count.max(0) as u32,
        max_threads_per_sm: max_threads_per_sm.max(0) as u32,
        rpx_regs_per_thread,
        rpx_blocks_per_sm,
        rpx_block_dim: RPX_BLOCK_DIM,
    })
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
    search(
        Arm::Keccak256,
        inner_lanes,
        grinding_factor,
        knobs_in_effect(),
    )
}

/// [`generate_nonce_gpu`] at knobs given here rather than read from the
/// environment.
///
/// ⛔ Not a second policy — the SAME [`search`], with the one thing the
/// environment would have decided passed in. It exists because the knobs are
/// cached in a `OnceLock`, so a process cannot compare two settings through the
/// environment: a sweep would need one process per arm, and two processes are
/// two device contexts, two cubin loads and two clock domains. The arms that
/// matter (the nonce is unmoved by the block size; what a grind costs at each
/// setting) are exact only when they share a process.
pub fn generate_nonce_gpu_at(
    inner_lanes: &[u64; 4],
    grinding_factor: u8,
    knobs: Knobs,
) -> Option<u64> {
    search(Arm::Keccak256, inner_lanes, grinding_factor, knobs)
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
    search(Arm::Rpx256, inner_felts, grinding_factor, knobs_in_effect())
}

/// [`generate_nonce_rpx_gpu`] at knobs given here rather than read from the
/// environment. See [`generate_nonce_gpu_at`] for why this exists.
pub fn generate_nonce_rpx_gpu_at(
    inner_felts: &[u64; 4],
    grinding_factor: u8,
    knobs: Knobs,
) -> Option<u64> {
    search(Arm::Rpx256, inner_felts, grinding_factor, knobs)
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
fn search(arm: Arm, inner: &[u64; 4], grinding_factor: u8, knobs: Knobs) -> Option<u64> {
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

    let count = knobs.block(grinding_factor);

    let cfg = LaunchConfig {
        grid_dim: (knobs.grid, 1, 1),
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

/// ⛔ DIAGNOSTIC: what one grind EXECUTED, beside what it returned.
///
/// Every field is counted ON THE DEVICE by the threads that did the work, so
/// `executed - (h + stride)` is a READ rather than a model. See
/// [`search_counted`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GrindCounts {
    /// The nonce the search returned — must equal the shipped kernel's.
    pub nonce: u64,
    /// Permutations every thread of every launch actually ran.
    pub executed: u64,
    /// The most iterations any single thread ran, over every launch.
    pub max_iters: u64,
    /// Threads that left by the loop bound rather than by the early exit.
    pub ran_to_end: u64,
    /// Launches this search made: the hit's block, plus every miss before it.
    pub launches: u64,
    /// The poll period (`k`) this search launched the twin with, echoed back
    /// from the launch so a report quotes the knob READ on the path rather than
    /// the one the caller believes it passed. `1` is the shipped
    /// every-iteration poll; the sweep uses powers of two.
    pub poll_period: u64,
}

/// ⛔ THE DIAGNOSTIC TWIN OF [`search`], RPX ONLY, ON NO PROVING PATH.
///
/// Identical walk, identical exits, identical `atomicMin` — plus three device
/// counters. It exists to separate two explanations of the same slow launch
/// that no stopwatch can tell apart: MORE PERMUTATIONS (a thread that did not
/// observe the early exit and kept hashing) against THE SAME PERMUTATIONS MORE
/// SLOWLY (a cost outside this loop entirely).
///
/// ⚠ ITS OWN ADMISSIBILITY IS THE CALLER'S JOB, and it is not optional: a
/// counted kernel with different register pressure has different occupancy and
/// therefore measures a different kernel. The bench asserts this twin
/// reproduces the shipped kernel's milliseconds per seed within the measured
/// noise floor, and reports the instrument as having changed the phenomenon if
/// it does not.
///
/// The counters accumulate ACROSS the launches of one search — `atomicAdd` for
/// the two sums, `atomicMax` for the deepest thread — so a miss-and-relaunch
/// seed reports the whole search and not only its last block. `launches` says
/// how many blocks that was.
///
/// `poll_period` sets how often each thread polls `*result` for the early
/// exit: every iteration at `1` (the shipped kernel's rate), every `k`-th at
/// `k`, staggered across threads so the request rate falls by `k` in both
/// average and peak. It exists to test whether the stale-poll overrun is
/// contention on that one address — lower the rate and read whether the overrun
/// falls (contention) or rises (a poll that is simply too coarse). It must be a
/// power of two; it changes only how much a thread over-scans, never the nonce
/// the search returns.
///
/// Returns `None` for the same reasons [`search`] does, plus a factor outside
/// the supported range or a `poll_period` that is zero or not a power of two.
pub fn search_counted(
    inner: &[u64; 4],
    grinding_factor: u8,
    knobs: Knobs,
    poll_period: u64,
) -> Option<GrindCounts> {
    if !(GRIND_MIN_FACTOR..=64).contains(&grinding_factor) {
        return None;
    }
    // The kernel derives its stagger mask as `poll_period - 1` and polls when
    // `(n + tid) & mask == 0`, which is a clean period of `poll_period` only
    // when `poll_period` is a power of two. `poll_period == 1` (mask 0) polls
    // every iteration — the shipped kernel's rate, the sweep's control; a zero
    // would wrap the mask to all-ones and poll essentially never. Reject both.
    if poll_period == 0 || !poll_period.is_power_of_two() {
        return None;
    }
    let limit: u64 = 1u64 << (64 - grinding_factor);

    let be = backend().ok()?;
    let stream = be.next_stream();
    let inner_dev = stream.clone_htod(inner.as_slice()).ok()?;

    let count = knobs.block(grinding_factor);
    let cfg = LaunchConfig {
        grid_dim: (knobs.grid, 1, 1),
        block_dim: (RPX_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };

    let sentinel = [u64::MAX];
    let mut result_dev = stream.clone_htod(&sentinel).ok()?;
    // One slot each, zeroed once and accumulated into by every launch.
    let zeros = [0u64; 3];
    let mut counts_dev = stream.clone_htod(&zeros).ok()?;

    let mut base: u64 = 0;
    let mut launches: u64 = 0;
    loop {
        stream.memcpy_htod(&sentinel, &mut result_dev).ok()?;
        launches += 1;
        // SAFETY: the same contract as `search`'s launch, with three more
        // device words the kernel only ever adds into or maxes against.
        unsafe {
            stream
                .launch_builder(&be.rpx_grind_search_counted)
                .arg(&inner_dev)
                .arg(&limit)
                .arg(&base)
                .arg(&count)
                .arg(&mut result_dev)
                .arg(&mut counts_dev)
                .arg(&poll_period)
                .launch(cfg)
                .ok()?;
        }
        let host = stream.clone_dtoh(&result_dev).ok()?;
        stream.synchronize().ok()?;
        if host[0] != u64::MAX {
            let counts = stream.clone_dtoh(&counts_dev).ok()?;
            stream.synchronize().ok()?;
            return Some(GrindCounts {
                nonce: host[0],
                executed: counts[0],
                max_iters: counts[1],
                ran_to_end: counts[2],
                launches,
                poll_period,
            });
        }
        base = base.checked_add(count)?;
    }
}

#[cfg(test)]
mod tests {
    //! The knobs' parser and their launch arithmetic, card-free: nothing here
    //! touches `backend()`, so these run wherever the crate compiles.
    //!
    //! `knobs_in_effect()` itself is deliberately NOT exercised — it caches in
    //! a `OnceLock`, so a test that set a variable would fix the value for
    //! every other test in the binary and the second assertion would pass on
    //! the first one's cache. That is also why the sweep goes through
    //! `generate_nonce_rpx_gpu_at` rather than through the environment.

    use super::{
        BLOCK_DIM, GRID_DEFAULT, GRID_RANGE, Knobs, MAX_BLOCK, MIN_BLOCK, RPX_BLOCK_DIM,
        SCAN_FACTOR_DEFAULT, SCAN_FACTOR_RANGE, parse_knob,
    };

    /// A knob at a named scan factor, the grid left at the record posture.
    fn at_scan(scan: u32) -> Knobs {
        Knobs {
            scan,
            grid: GRID_DEFAULT,
        }
    }

    /// Every arm of the scan sweep parses, and so does every arm of the grid
    /// sweep — both directions, since the grid's optimum can sit on either side
    /// of the default.
    #[test]
    fn the_parser_accepts_every_arm_of_both_sweeps() {
        for k in [8u32, 4, 2, 1] {
            assert_eq!(
                parse_knob(&k.to_string(), &SCAN_FACTOR_RANGE),
                Some(k),
                "scan factor {k} is a sweep arm and must parse"
            );
        }
        for g in [256u32, 512, 1024, 2048, 4096] {
            assert_eq!(
                parse_knob(&g.to_string(), &GRID_RANGE),
                Some(g),
                "grid {g} is a sweep arm and must parse"
            );
        }
        assert_eq!(parse_knob("64", &SCAN_FACTOR_RANGE), Some(64), "scan top");
        assert_eq!(parse_knob("65536", &GRID_RANGE), Some(65_536), "grid top");
    }

    /// The refusing half, executed on both ranges. Each of these would
    /// otherwise reach the launch and become a DIFFERENT search that no log
    /// could distinguish from the one the operator asked for.
    #[test]
    fn the_parser_refuses_what_would_silently_change_the_search() {
        for raw in ["0", "abc", "", "-1", "8.0", "0x8", " ", "4294967296"] {
            assert_eq!(
                parse_knob(raw, &SCAN_FACTOR_RANGE),
                None,
                "scan {raw:?} must be refused, not defaulted"
            );
            assert_eq!(
                parse_knob(raw, &GRID_RANGE),
                None,
                "grid {raw:?} must be refused, not defaulted"
            );
        }
        // Each range refuses just past its own top, and the two tops differ —
        // so a range accidentally shared between the knobs reddens here.
        assert_eq!(parse_knob("65", &SCAN_FACTOR_RANGE), None, "scan past top");
        assert_eq!(parse_knob("65537", &GRID_RANGE), None, "grid past top");
        assert_eq!(
            parse_knob("1024", &SCAN_FACTOR_RANGE),
            None,
            "a grid value is not a scan factor"
        );
    }

    /// The block size at the production grinding factor, written out as
    /// arithmetic rather than as the expression under test.
    ///
    /// ⇒ A drifted default or a dropped multiply reddens here by value.
    #[test]
    fn the_block_is_the_hit_distance_times_the_scan_factor() {
        assert_eq!(at_scan(8).block(20), 8 * 1_048_576, "2^20 * 8 = 2^23");
        assert_eq!(at_scan(4).block(20), 4 * 1_048_576, "2^20 * 4 = 2^22");
        assert_eq!(at_scan(2).block(20), 2 * 1_048_576, "2^20 * 2 = 2^21");
        assert_eq!(at_scan(1).block(20), 1_048_576, "2^20 * 1 = 2^20");
    }

    /// ★ The stride — the term the kernels' early exit leaves behind as
    /// overshoot, and the whole mechanism of the grid knob. Written as the
    /// products themselves, so a changed block dim reddens rather than being
    /// absorbed by the formula under test.
    #[test]
    fn the_stride_is_the_grid_times_the_block_dim() {
        assert_eq!(RPX_BLOCK_DIM, 128, "the RPX arm's block dim");
        assert_eq!(BLOCK_DIM, 256, "the keccak arm's block dim");
        assert_eq!(
            Knobs::DEFAULT.stride(RPX_BLOCK_DIM),
            131_072,
            "1024 blocks * 128 threads"
        );
        assert_eq!(
            Knobs::DEFAULT.stride(BLOCK_DIM),
            262_144,
            "1024 blocks * 256 threads"
        );
        // Both directions of the sweep, as products.
        for (grid, rpx) in [
            (256u32, 32_768u64),
            (512, 65_536),
            (1024, 131_072),
            (2048, 262_144),
            (4096, 524_288),
        ] {
            let knobs = Knobs { scan: 8, grid };
            assert_eq!(
                knobs.stride(RPX_BLOCK_DIM),
                rpx,
                "grid {grid} on the RPX arm"
            );
        }
    }

    /// The defaults are the record posture: every measurement in the campaign
    /// was taken at a 2^23 block over a 1024-block grid, and only an ABBA may
    /// move either.
    #[test]
    fn the_defaults_are_the_record_posture() {
        assert_eq!(SCAN_FACTOR_DEFAULT, 8, "the record posture's scan factor");
        assert_eq!(GRID_DEFAULT, 1024, "the record posture's grid");
        assert_eq!(Knobs::DEFAULT.scan, 8, "DEFAULT carries the scan factor");
        assert_eq!(Knobs::DEFAULT.grid, 1024, "DEFAULT carries the grid");
        assert_eq!(
            Knobs::DEFAULT.block(20),
            1 << 23,
            "the record posture's per-launch block at grinding factor 20"
        );
    }

    /// Both ends of the clamp still bind, including the u64 overflow the
    /// saturating multiply exists for. The grid does not enter the clamp at
    /// all — it sizes the stride, not the block — and that separation is the
    /// thing this asserts.
    #[test]
    fn the_clamp_binds_at_both_ends_and_the_grid_does_not_touch_it() {
        assert_eq!(
            at_scan(1).block(12),
            MIN_BLOCK,
            "the min-factor gate's own factor, at the narrowest scan, floors"
        );
        assert_eq!(at_scan(8).block(26), MAX_BLOCK, "2^26 * 8 = 2^29 ceils");
        assert_eq!(
            at_scan(8).block(64),
            MAX_BLOCK,
            "2^64 saturates before the multiply, then ceils"
        );
        assert!(
            (MIN_BLOCK..=MAX_BLOCK).contains(&Knobs::DEFAULT.block(20)),
            "the record posture sits strictly inside the clamp, so neither end \
             is silently setting it"
        );
        for grid in [256u32, 1024, 4096] {
            assert_eq!(
                Knobs { scan: 8, grid }.block(20),
                1 << 23,
                "the grid must not move the per-launch block"
            );
        }
    }
}
