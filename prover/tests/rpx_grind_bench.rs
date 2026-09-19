//! ★ What one device grind costs, and what the two launch knobs do to it.
//!
//! Round 2's lever was ruled from `math-cuda/src/grinding.rs`: one launch sizes
//! its block at 8x the expected hit distance, so "drop the 8 and drop 8x the
//! permutations". The kernels say otherwise — both carry
//! `if (nonce >= *result) break;` against a `volatile` result the `atomicMin`
//! writes through L2, and the stride walk gives every nonce in the block
//! exactly one owner, so the scan STOPS at the first hit. The executed
//! permutations are `h + stride` at ANY scan factor; the block size is a
//! ceiling the launch never reaches, and `stride = grid x block_dim` is the
//! term that is left.
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda \
//!     --test rpx_grind_bench -- --ignored --nocapture
//! ```
//!
//! ONE process, every arm, and that is deliberate: the knobs cache in a
//! `OnceLock`, so reading two settings through the environment would need two
//! processes — two device contexts, two cubin loads, two clock domains. Every
//! arm here runs on the SAME seeds through `generate_nonce_rpx_gpu_at`, so the
//! comparison is PAIRED on identical hit distances and the seed-to-seed spread
//! (which is exponential, and wide) cancels between arms instead of being
//! averaged away. The environment path is exercised too, and asserted to agree.
//!
//! WHAT TO READ, pre-registered:
//!
//! * **the scan arms** (8/4/2/1 at grid 1024) — FLAT means the block size is a
//!   ceiling and the ruled lever is dead; halving with the factor means the
//!   scan dominates and it is real. The k=1 arm prices the extra round trips
//!   (1.58 expected launches per grind against 1.00).
//! * **the grid arms** (256/512/1024/2048/4096 at scan 8) — the sweep runs BOTH
//!   ways because the knob has two edges: under-filled, a wider grid raises
//!   throughput faster than overshoot and the wall falls; filled, extra blocks
//!   only queue and the wider stride is pure added work. `DEVICE FILL` below
//!   says which edge the default sits on, READ from the driver.
//! * **`ns/perm`** — the invariant. Time is about `(h/stride + 1)` rounds, so
//!   `ms/grind` carries the seed's hit distance while `ns/perm` does not; the
//!   grid question is a throughput question and this is the throughput.
//! * **the nonce lists** — identical across every arm, or the search is not
//!   scanning contiguously from zero and the geometry is moving the answer.
//! * **the CALIBRATION CONTROL** — this bench's ms/grind at the record posture
//!   times the base's 3,428 grinds, against the base's grind wall read from
//!   wt14. Out of that window and nothing here describes the block's grind.
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

/// Grinds per arm, all arms on the same seeds. The hit distance is exponential
/// with mean `2^20`, so a single seed says nothing; pairing across arms is what
/// makes the ratios precise, and this many keeps the unpaired mean's standard
/// error near 6%.
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

/// One arm's reading.
struct Arm {
    knobs: Knobs,
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    /// Time divided by the permutations actually executed, `h + stride`. The
    /// returned nonce IS `h`, so this is measured rather than modelled.
    ns_per_perm: f64,
    nonces: Vec<u64>,
}

#[test]
#[ignore = "device benchmark; run with --ignored --nocapture on the GPU box"]
fn what_one_grind_costs_at_each_launch_geometry() {
    let seeds: Vec<[u8; 32]> = (0..RUNS).map(seed_for).collect();

    print_device_fill();

    // Warm-up, excluded by name: the first launch in a process pays context
    // creation and the cubin load, which is not what a grind costs in a proof
    // that has already done three thousand of them.
    let warm = one_grind(&seeds[0], Knobs::DEFAULT);
    println!(
        "GRIND BENCH: warm-up nonce {} in {:.3} ms (EXCLUDED)",
        warm.0, warm.1
    );

    // The environment path, exercised once so the knob is shown READ on the
    // same path the tree runs use — and its answer tied to the explicit path.
    let env_knobs = math_cuda::grinding::knobs_in_effect();
    let env_nonce = math_cuda::grinding::generate_nonce_rpx_gpu(
        &inner_hash_felts::<RpxGrind>(&seeds[0], GRINDING_FACTOR),
        GRINDING_FACTOR,
    )
    .expect("GPU RPX grind through the environment path (needs a GPU)");
    let explicit_nonce = one_grind(&seeds[0], env_knobs).0;
    assert_eq!(
        env_nonce, explicit_nonce,
        "the environment path and the explicit path disagree at {env_knobs:?} — \
         one of them is not the search the other is"
    );
    println!("GRIND BENCH: environment path agrees with the explicit path at {env_knobs:?}");

    let scan_arms: Vec<Arm> = [8u32, 4, 2, 1]
        .into_iter()
        .map(|scan| run_arm(Knobs { scan, grid: 1024 }, &seeds))
        .collect();
    let grid_arms: Vec<Arm> = [256u32, 512, 2048, 4096]
        .into_iter()
        .map(|grid| run_arm(Knobs { scan: 8, grid }, &seeds))
        .collect();

    // The record posture is the shared arm: the head of the scan sweep and the
    // 1024 point of the grid sweep are the same measurement, not two.
    let record = &scan_arms[0];

    println!("\n=== THE ARMS (paired: every arm ran the same {RUNS} seeds) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "knobs", "mean ms", "median", "min", "max", "ns/perm", "vs 8/1024"
    );
    print_arm(record, record);
    println!("-- the scan sweep, grid 1024 (pre-registered FLAT if the block is a ceiling) --");
    for arm in &scan_arms[1..] {
        print_arm(arm, record);
    }
    println!("-- the grid sweep, scan 8 (pre-registered to fall on ONE side of 1024, not both) --");
    for arm in &grid_arms {
        print_arm(arm, record);
    }

    // ★ THE NONCE CONTROL. Every arm must have returned the same answer for the
    // same seed; the launch geometry cannot move it, and that is why sweeping
    // either knob moves no proof byte.
    for arm in scan_arms.iter().chain(&grid_arms) {
        assert_eq!(
            arm.nonces, record.nonces,
            "{:?} returned a different nonce list from the record posture — the \
             launch geometry moved the answer, so the search is not scanning \
             contiguously from zero and nothing else here is quotable",
            arm.knobs
        );
    }
    println!(
        "\nNONCE CONTROL: all {} arms returned identical nonce lists over {RUNS} seeds",
        scan_arms.len() + grid_arms.len()
    );

    // ★ THE CALIBRATION CONTROL, said in or out.
    let projected = record.mean_ms * BASE_GRINDS / 1000.0;
    let inside = (CALIBRATION_LOW_MS..=CALIBRATION_HIGH_MS).contains(&record.mean_ms);
    println!("\n=== THE CALIBRATION CONTROL ===");
    println!(
        "this bench at the record posture: {:.3} ms/grind; the block's window \
         {CALIBRATION_LOW_MS:.3}-{CALIBRATION_HIGH_MS:.3} ms/grind \
         (15.73-18.03 s over {BASE_GRINDS:.0} grinds, wt14)",
        record.mean_ms
    );
    println!(
        "projected over the base's grinds: {projected:.2} s against 15.73-18.03 s  =>  {}",
        if inside {
            "IN — this bench measures the block's grind"
        } else {
            "OUT — this bench is NOT the block's grind and nothing above is quotable"
        }
    );
}

/// One arm: every seed, one process, explicit knobs.
fn run_arm(knobs: Knobs, seeds: &[[u8; 32]]) -> Arm {
    let stride = knobs.stride(math_cuda::grinding::RPX_BLOCK_DIM) as f64;
    let mut times = Vec::with_capacity(seeds.len());
    let mut nonces = Vec::with_capacity(seeds.len());
    let mut total_ns = 0.0f64;
    let mut total_perms = 0.0f64;
    for seed in seeds {
        let (nonce, ms) = one_grind(seed, knobs);
        total_ns += ms * 1.0e6;
        // The returned nonce IS the first valid one, so the permutations the
        // launch executed are `nonce + stride` — measured, not modelled.
        total_perms += nonce as f64 + stride;
        times.push(ms);
        nonces.push(nonce);
    }
    let mean_ms = times.iter().sum::<f64>() / times.len() as f64;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing"));
    Arm {
        knobs,
        mean_ms,
        median_ms: sorted[sorted.len() / 2],
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
        ns_per_perm: total_ns / total_perms,
        nonces,
    }
}

/// One grind at the production factor, timed around the device call alone.
///
/// The nonce is re-validated on the host exactly as the prover's dispatch does,
/// so a timing arm cannot quietly become a measurement of a kernel that returns
/// garbage quickly. A device search that failed returns `None` and takes the
/// `expect`, so every timed call is a real device grind.
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
                "SMs {} · max threads/SM {} · rpx grind kernel: {} regs/thread, \
                 block dim {}, {} resident blocks/SM",
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
            println!("{:<8} {:>12} {:>10}", "grid", "fill vs ceiling", "stride");
            for grid in [256u32, 512, 1024, 2048, 4096] {
                println!(
                    "{:<8} {:>11.2}x {:>10}",
                    grid,
                    fill.fill(grid),
                    Knobs { scan: 8, grid }.stride(fill.rpx_block_dim)
                );
            }
            println!(
                "⇒ a grid above {} queues: the surplus blocks buy no parallelism \
                 and their stride is pure overshoot past the first hit",
                fill.resident_blocks()
            );
        }
    }
}

fn print_arm(arm: &Arm, record: &Arm) {
    println!(
        "scan {:<3} grid {:<5} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.2} {:>8.3}",
        arm.knobs.scan,
        arm.knobs.grid,
        arm.mean_ms,
        arm.median_ms,
        arm.min_ms,
        arm.max_ms,
        arm.ns_per_perm,
        arm.mean_ms / record.mean_ms
    );
}
