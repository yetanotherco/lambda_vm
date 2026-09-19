//! ★ What one device grind costs, and what the two launch knobs do to it.
//!
//! # v3 — the order control, and the one split that can falsify a mechanism
//!
//! v2 ran its arms in one fixed order inside one process and read a monotone
//! fall down the SCAN column — ratios of 1.000, 0.953, 0.872 and 0.800 at scan
//! factors 8, 4, 2 and 1. That contradicts the kernels, which both carry
//! `if (nonce >= *result) break;` against a `volatile` result the `atomicMin`
//! writes through L2, so once the first valid nonce `h` is found every thread
//! stops within one stride round and the executed permutations are `h + stride`
//! WHATEVER the block size. For the median seed, whose `h` is below even the
//! narrowest block here, the two launches are the same kernel doing the same
//! rounds — there is nothing for the knob to change.
//!
//! Pairing on seeds cancels the seed spread. It does not cancel DRIFT, and a
//! monotone fall in run order is what a boosting clock looks like. So:
//!
//! 1. **Every seed runs EVERY arm, in a rotating order** (seed `s` starts at arm
//!    `s % N`). Drift now pairs out too: each arm sits in every position of the
//!    rotation equally often.
//! 2. **The control is repeated as the LAST arm.** Identical knobs to arm 0, so
//!    its ratio must read ≈ 1.00. It is the noise floor of the whole procedure,
//!    measured rather than assumed; anything the other arms claim must clear it.
//! 3. **Per-seed PAIRED statistics** — the median of the per-seed ratios and the
//!    COUNT of seeds on which the arm beat the control. A real 20% shows on most
//!    of 256 seeds; drift shows as a trend that a rotation destroys.
//! 4. ⭐ **THE SPLIT THAT CAN FALSIFY A MECHANISM.** The returned nonce IS `h`,
//!    so each seed can be labelled by whether `h` fell inside the arm's block.
//!    Seeds with `h < block` take ONE launch at every arm and run the identical
//!    kernel — no mechanism can touch them. Seeds with `h >= block` are the only
//!    ones that miss and relaunch. So if an arm's gain is the same on both
//!    groups, it is NOT the knob; if it lives entirely in the `h >= block`
//!    group, there is a real miss-path effect to explain.
//! 5. The COMBINED arms, in case the two effects are real and additive.
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda \
//!     --test rpx_grind_bench -- --ignored --nocapture
//! ```
//!
//! Lives in the prover crate rather than `math-cuda` for the same reason
//! `rpx_device_parity.rs` does: the host side (`RpxStarkHash`) lives here, and
//! `math-cuda` is a dev-dependency of this crate, not the reverse. RPX because
//! that is the hash the record posture grinds under. Needs a GPU.
#![cfg(feature = "cuda")]

use std::time::Instant;

use lambda_vm_prover::lfm::algebraic_commit::RpxStarkHash;
use math_cuda::grinding::Knobs;
use stark::config::GrindingDigest;
use stark::grinding::{inner_hash_felts, is_valid_nonce};

/// The digest the RPX configuration grinds over — its transcript's hash.
type RpxGrind = GrindingDigest<RpxStarkHash>;

/// The production grinding factor. Not a knob here: the bit count is the
/// security parameter, and this measurement is about the launch, not the bits.
const GRINDING_FACTOR: u8 = 20;

/// Grinds per arm, every arm on the same seeds.
const RUNS: usize = 256;

/// ★ THE CALIBRATION WINDOW, from the block and not from this bench.
///
/// The base performs 3,428 device grinds (lb19/lb20: `rpx grinds 3428`, and
/// `states 3428 = grinds`). wt14's grind wall over the base is 14.84 s in the
/// 15 epochs' group openings plus 0.89 s in the global, plus the prepared
/// openings' share which is not separately measured and is bounded by
/// `open_prepared` = 2.3 s. So 15.73-18.03 s over 3,428 grinds.
const BASE_GRINDS: f64 = 3428.0;
const CALIBRATION_LOW_MS: f64 = 15.73 * 1000.0 / BASE_GRINDS;
const CALIBRATION_HIGH_MS: f64 = 18.03 * 1000.0 / BASE_GRINDS;

/// The arms, in DEFINITION order. The rotation decides RUN order.
///
/// Arm 0 is the control and the record posture. The last arm repeats it
/// byte-for-byte: same knobs, different position in every rotation, so its
/// ratio is the procedure's own noise floor.
fn arms() -> Vec<(&'static str, Knobs)> {
    vec![
        (
            "control 8/1024",
            Knobs {
                scan: 8,
                grid: 1024,
            },
        ),
        (
            "scan 4",
            Knobs {
                scan: 4,
                grid: 1024,
            },
        ),
        (
            "scan 2",
            Knobs {
                scan: 2,
                grid: 1024,
            },
        ),
        (
            "scan 1",
            Knobs {
                scan: 1,
                grid: 1024,
            },
        ),
        ("grid 256", Knobs { scan: 8, grid: 256 }),
        ("grid 512", Knobs { scan: 8, grid: 512 }),
        (
            "grid 2048",
            Knobs {
                scan: 8,
                grid: 2048,
            },
        ),
        (
            "grid 4096",
            Knobs {
                scan: 8,
                grid: 4096,
            },
        ),
        ("scan 1 grid 512", Knobs { scan: 1, grid: 512 }),
        ("scan 1 grid 256", Knobs { scan: 1, grid: 256 }),
        (
            "control AGAIN",
            Knobs {
                scan: 8,
                grid: 1024,
            },
        ),
    ]
}

#[test]
#[ignore = "device benchmark; run with --ignored --nocapture on the GPU box"]
fn what_one_grind_costs_at_each_launch_geometry() {
    let arms = arms();
    let n = arms.len();
    let seeds: Vec<[u8; 32]> = (0..RUNS).map(seed_for).collect();

    print_device_fill();

    // Warm-up, excluded by name: the first launch in a process pays context
    // creation and the cubin load.
    let warm = one_grind(&seeds[0], Knobs::DEFAULT);
    println!(
        "GRIND BENCH: warm-up nonce {} in {:.3} ms (EXCLUDED)",
        warm.0, warm.1
    );

    // The environment path, exercised once and tied to the explicit path, so
    // the knob is shown READ on the same path the tree runs use.
    let env_knobs = math_cuda::grinding::knobs_in_effect();
    let env_nonce = math_cuda::grinding::generate_nonce_rpx_gpu(
        &inner_hash_felts::<RpxGrind>(&seeds[0], GRINDING_FACTOR),
        GRINDING_FACTOR,
    )
    .expect("GPU RPX grind through the environment path (needs a GPU)");
    assert_eq!(
        env_nonce,
        one_grind(&seeds[0], env_knobs).0,
        "the environment path and the explicit path disagree at {env_knobs:?}"
    );
    println!("GRIND BENCH: environment path agrees with the explicit path at {env_knobs:?}");

    // ── THE RUN: seeds outer, arms inner, ROTATED ──────────────────────────
    let mut times = vec![vec![0.0f64; RUNS]; n];
    let mut nonces = vec![vec![0u64; RUNS]; n];
    for (s, seed) in seeds.iter().enumerate() {
        for j in 0..n {
            let a = (s + j) % n;
            let (nonce, ms) = one_grind(seed, arms[a].1);
            times[a][s] = ms;
            nonces[a][s] = nonce;
        }
    }
    println!(
        "GRIND BENCH: {} arms x {RUNS} seeds, rotated (seed s starts at arm s % {})",
        n, n
    );

    // ★ THE NONCE CONTROL, first: if the geometry moved the answer, nothing
    // else here means anything.
    for (a, (name, knobs)) in arms.iter().enumerate() {
        assert_eq!(
            nonces[a], nonces[0],
            "{name} ({knobs:?}) returned a different nonce list from the control \
             — the launch geometry moved the answer, so the search is not \
             scanning contiguously from zero"
        );
    }
    println!("NONCE CONTROL: all {n} arms returned identical nonce lists over {RUNS} seeds");

    // ── THE TABLE ──────────────────────────────────────────────────────────
    println!("\n=== THE ARMS (paired per seed, rotated order) ===");
    println!(
        "{:<18} {:>9} {:>9} {:>9} {:>11} {:>10} {:>9} {:>9}",
        "arm", "mean ms", "median", "ns/perm", "median rat", "beat/256", "rat h<blk", "rat h>=blk"
    );
    for (a, (name, knobs)) in arms.iter().enumerate() {
        print_arm(name, *knobs, &times[a], &times[0], &nonces[0]);
    }

    // ── THE CALIBRATION CONTROL ────────────────────────────────────────────
    let control_mean = mean(&times[0]);
    let projected = control_mean * BASE_GRINDS / 1000.0;
    let inside = (CALIBRATION_LOW_MS..=CALIBRATION_HIGH_MS).contains(&control_mean);
    println!("\n=== THE CALIBRATION CONTROL ===");
    println!(
        "this bench at the record posture: {control_mean:.3} ms/grind; the block's window \
         {CALIBRATION_LOW_MS:.3}-{CALIBRATION_HIGH_MS:.3} ms/grind (15.73-18.03 s over \
         {BASE_GRINDS:.0} grinds, wt14)"
    );
    println!(
        "projected over the base's grinds: {projected:.2} s against 15.73-18.03 s  =>  {}",
        if inside {
            "IN — this bench measures the block's grind"
        } else {
            "OUT — this bench is NOT the block's grind and nothing above is quotable"
        }
    );

    println!("\n=== HOW TO READ IT (pre-registered) ===");
    println!(
        "  `control AGAIN` is the NOISE FLOOR: its median ratio must read ~1.00. Any arm \
         claiming less than that floor is claiming noise."
    );
    println!(
        "  ⭐ `rat h<blk` vs `rat h>=blk`: seeds whose hit fell INSIDE the arm's block take \
         ONE launch and run the identical kernel at every arm, so no knob can touch them. \
         An arm whose gain is the SAME on both groups is not the knob — it is the procedure. \
         A gain living only in `h>=blk` is a real miss-path effect and owes a mechanism."
    );
}

/// One arm's row, paired against the control seed by seed.
fn print_arm(name: &str, knobs: Knobs, t: &[f64], ctl: &[f64], nonces: &[u64]) {
    let block = knobs.block(GRINDING_FACTOR);
    let ratios: Vec<f64> = t.iter().zip(ctl).map(|(a, b)| a / b).collect();
    let beat = ratios.iter().filter(|r| **r < 1.0).count();
    let inside: Vec<f64> = ratios
        .iter()
        .zip(nonces)
        .filter(|(_, h)| **h < block)
        .map(|(r, _)| *r)
        .collect();
    let outside: Vec<f64> = ratios
        .iter()
        .zip(nonces)
        .filter(|(_, h)| **h >= block)
        .map(|(r, _)| *r)
        .collect();
    let stride = knobs.stride(math_cuda::grinding::RPX_BLOCK_DIM) as f64;
    let perms: f64 = nonces.iter().map(|h| *h as f64 + stride).sum();
    println!(
        "{name:<18} {:>9.3} {:>9.3} {:>9.2} {:>11.3} {:>6}/{:<3} {:>9} {:>10}",
        mean(t),
        median(t),
        t.iter().sum::<f64>() * 1.0e6 / perms,
        median(&ratios),
        beat,
        ratios.len(),
        fmt_group(&inside),
        fmt_group(&outside),
    );
}

/// A group's median ratio and its size, or a dash when the group is empty.
fn fmt_group(rs: &[f64]) -> String {
    if rs.is_empty() {
        "—".to_string()
    } else {
        format!("{:.3}/{}", median(rs), rs.len())
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing"));
    v[v.len() / 2]
}

/// One grind at the production factor, timed around the device call alone.
///
/// The nonce is re-validated on the host exactly as the prover's dispatch does,
/// so a timing arm cannot quietly become a measurement of a kernel that returns
/// garbage quickly.
fn one_grind(seed: &[u8; 32], knobs: Knobs) -> (u64, f64) {
    let felts = inner_hash_felts::<RpxGrind>(seed, GRINDING_FACTOR);
    let started = Instant::now();
    let nonce = math_cuda::grinding::generate_nonce_rpx_gpu_at(&felts, GRINDING_FACTOR, knobs)
        .expect("GPU RPX grind (needs a GPU)");
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        is_valid_nonce::<RpxGrind>(seed, nonce, GRINDING_FACTOR),
        "GPU nonce {nonce} from {knobs:?} fails is_valid_nonce — a timing number \
         from an invalid nonce is not a measurement"
    );
    (nonce, ms)
}

/// Distinct seeds, so each arm pays its own spread of hit distances.
fn seed_for(i: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&(i as u64).to_le_bytes());
    seed[8] = 0xA5;
    seed
}

/// ★ What the driver says about filling this card — every field READ, so the
/// residency question the grid knob turns on is answered rather than estimated.
fn print_device_fill() {
    println!("\n=== DEVICE FILL (read from the driver, not estimated) ===");
    match math_cuda::grinding::device_fill() {
        None => println!("DEVICE FILL: unavailable — no device, or the driver refused the query"),
        Some(fill) => {
            println!(
                "SMs {} · max threads/SM {} · rpx grind kernel: {} regs/thread, block dim {}, \
                 {} resident blocks/SM",
                fill.sm_count,
                fill.max_threads_per_sm,
                fill.rpx_regs_per_thread,
                fill.rpx_block_dim,
                fill.rpx_blocks_per_sm
            );
            println!(
                "⇒ resident ceiling: {} blocks = {} threads",
                fill.resident_blocks(),
                fill.resident_threads()
            );
            println!("{:<8} {:>16} {:>10}", "grid", "fill vs ceiling", "stride");
            for grid in [256u32, 512, 1024, 2048, 4096] {
                println!(
                    "{:<8} {:>15.2}x {:>10}",
                    grid,
                    fill.fill(grid),
                    Knobs { scan: 8, grid }.stride(fill.rpx_block_dim)
                );
            }
            println!(
                "⇒ a grid above {} queues: the surplus blocks buy no parallelism and their \
                 stride is pure overshoot past the first hit",
                fill.resident_blocks()
            );
        }
    }
}
