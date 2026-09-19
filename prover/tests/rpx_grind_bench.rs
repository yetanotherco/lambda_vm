//! ★ What one device grind costs, and what the scan factor does to it.
//!
//! Round 2's lever was ruled from `math-cuda/src/grinding.rs` — one launch
//! sizes its block at 8x the expected hit distance — on the reading that the
//! 8x is 8x the permutations. The kernels say otherwise: both carry
//! `if (nonce >= *result) break;` against a `volatile` result the `atomicMin`
//! writes through L2, and the stride walk gives every nonce in the block
//! exactly one owner, so the scan STOPS at the first hit and the factor is a
//! ceiling the launch never reaches.
//!
//! This is the arm that settles it on the card rather than from the source, in
//! seconds rather than in four tree runs:
//!
//! ```text
//! LAMBDA_VM_GRIND_SCAN_FACTOR=8 cargo test -p lambda-vm-prover --release \
//!     --features cuda --test rpx_grind_bench -- --ignored --nocapture
//! ```
//!
//! run once per factor in 8 / 4 / 2 / 1. Read ACROSS the arms:
//!
//! * **ms/grind** — flat across the four ⇒ the block size is a ceiling and the
//!   ruled lever is dead; halving with the factor ⇒ the scan dominates and it
//!   is real. The k = 1 arm additionally prices the extra round trips (1.58
//!   expected launches per grind against 1.00 at k = 8).
//! * **the nonces** — IDENTICAL across the four, or the search is not scanning
//!   contiguously from zero and the block size is moving the answer. The nonce
//!   is a function of the inner hash and the grinding factor alone; this is the
//!   control that says so, and it is also why the sweep moves no proof byte.
//!
//! Lives in the prover crate rather than `math-cuda` for the same reason
//! `rpx_device_parity.rs` does: the host side — `RpxStarkHash` — lives here,
//! and `math-cuda` is a dev-dependency of this crate, not the reverse.
//!
//! RPX because that is the hash the record posture grinds under
//! (`LAMBDA_VM_WHIR_HASH=rpx`). Needs a GPU.
#![cfg(feature = "cuda")]

use std::time::Instant;

use lambda_vm_prover::lfm::algebraic_commit::RpxStarkHash;
use stark::config::GrindingDigest;
use stark::grinding::{inner_hash_felts, is_valid_nonce};

/// The digest the RPX configuration grinds over — its transcript's hash.
type RpxGrind = GrindingDigest<RpxStarkHash>;

/// The production grinding factor. Not a knob here: the bit count is the
/// security parameter, and this measurement is about the block, not the bits.
const GRINDING_FACTOR: u8 = 20;

/// Grinds timed per arm. Each uses a distinct seed, so each pays its own hit
/// distance and the mean is taken over the geometric spread rather than over
/// one lucky nonce repeated.
const RUNS: usize = 32;

#[test]
#[ignore = "device benchmark; run with --ignored --nocapture on the GPU box"]
fn what_one_grind_costs_at_this_scan_factor() {
    // Read through the production accessor, so the line below reports the knob
    // the search is actually using and not a second reading of the
    // environment. This also forces the `★ GRIND SCAN FACTOR:` banner.
    let scan = math_cuda::grinding::scan_factor_in_effect();
    let block = math_cuda::grinding::per_launch_block(GRINDING_FACTOR);
    let expected: u64 = 1 << GRINDING_FACTOR;

    println!("GRIND BENCH: rpx256 · grinding factor {GRINDING_FACTOR} · scan factor {scan}");
    println!(
        "GRIND BENCH: expected hit distance {expected} · per-launch block {block} ({:.2}x) \
         · expected launches per grind {:.3}",
        block as f64 / expected as f64,
        1.0 / (1.0 - (-(scan as f64)).exp())
    );

    // A warm-up grind, excluded from the statistics: the first launch in the
    // process pays context creation and the cubin load, which is not what a
    // grind costs in a proof that has already done three thousand of them.
    let (warm, warm_ms) = grind(0xFF);
    println!("GRIND BENCH: warm-up nonce {warm} {warm_ms:.3} ms (EXCLUDED)");

    let mut total = 0.0f64;
    let mut worst = 0.0f64;
    let mut best = f64::INFINITY;
    for i in 0..RUNS {
        let (nonce, ms) = grind(i as u8);
        total += ms;
        worst = worst.max(ms);
        best = best.min(ms);
        // Every nonce printed, because the cross-arm control is an equality
        // between two runs' nonce LISTS and a mean cannot carry it.
        println!("GRIND BENCH: seed {i:02} nonce {nonce} {ms:.3} ms");
    }

    let mean = total / RUNS as f64;
    println!(
        "GRIND BENCH RESULT: scan {scan} · {RUNS} grinds · mean {mean:.3} ms · min {best:.3} \
         · max {worst:.3}"
    );
    // Both competing models, printed, so the arm reading this log does not have
    // to recompute them — and so the comparison across arms stays a comparison
    // of MEASURED means rather than of a mean against a model.
    println!(
        "GRIND BENCH MODEL: if the block is scanned in full, {:.1} M perm/s; if the scan stops \
         at the first hit, {:.1} M perm/s",
        block as f64 / (mean / 1000.0) / 1e6,
        expected as f64 / (mean / 1000.0) / 1e6
    );
}

/// One grind at the production factor over a seed derived from `tag`, timed
/// around the device call alone.
///
/// The nonce is re-validated on the host exactly as the prover's dispatch
/// does, so a timing arm cannot quietly become a measurement of a kernel that
/// returns garbage quickly. A device search that failed would return `None`
/// here and take the `expect`, so every timed call is a real device grind.
fn grind(tag: u8) -> (u64, f64) {
    let seed = [tag; 32];
    let felts = inner_hash_felts::<RpxGrind>(&seed, GRINDING_FACTOR);
    let started = Instant::now();
    let nonce = math_cuda::grinding::generate_nonce_rpx_gpu(&felts, GRINDING_FACTOR)
        .expect("GPU RPX grind (needs a GPU)");
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        is_valid_nonce::<RpxGrind>(&seed, nonce, GRINDING_FACTOR),
        "GPU nonce {nonce} fails is_valid_nonce (factor {GRINDING_FACTOR}) — a timing number \
         from an invalid nonce is not a measurement"
    );
    (nonce, ms)
}
