//! ★ STAGE B: do the slow grind launches do MORE WORK, or the same work slower?
//!
//! # What the timings could not answer
//!
//! `rpx_grind_bench` reads a mean about 20% above the model at the record
//! posture while the MEDIAN seed sits exactly on it, so a minority of launches
//! carries a large absolute cost. Three candidates died to that bench: the
//! power limiter (the movers are SMALL-`h` seeds, not long sustained launches),
//! the miss-and-relaunch path (the movers take ONE launch at both arms) and the
//! volatile load's per-iteration cost (the same iterations either way).
//!
//! v5 swept the scan factor UPWARD to test the last one standing — threads
//! running to the loop bound. They do not: the excess SATURATES above `count`
//! = 2^23 instead of growing with it (`vs control` 1.05 at scan 16, 32 and 64
//! against a prediction of 2.14, 4.43 and 9.00).
//!
//! But it does not saturate immediately either. Converted to iterations per
//! thread, `N = count / stride`, the excess fits a bounded quantity approached
//! geometrically:
//!
//! ```text
//!   N    8     16     32     64    128    256    512
//!   ms   0   0.354  0.763  1.004  1.055  1.057  1.060
//!   excess = T·(1 − e^(−(N−8)/τ)),  T ≈ 1.06 ms,  τ ≈ 19 iterations
//! ```
//!
//! Three independent points agree on τ within 4%. That is the shape of a
//! thread that keeps scanning for a bounded TIME after the answer is known —
//! about 19 further iterations, a per-iteration stopping probability near 5% —
//! and NOT the shape of one running to the loop bound. One iteration at grid
//! 1024 is 131,072 permutations, ≈ 0.56 ms at the bench's own 4.27 ns/perm, so
//! 19 iterations is ≈ 10.6 ms, which is where v4's top movers sat (5-11 ms).
//!
//! ⚠ That is a three-point fit with two parameters, measured against a baseline
//! (scan 1, eight iterations) that may itself be carrying capped excess. It is
//! a hypothesis, not a reading. This file replaces it with a reading.
//!
//! # What this measures
//!
//! `rpx_grind_search_counted` counts, on the device, the permutations its
//! threads actually ran. So `executed − (h + stride)` is the overrun, per
//! search, as a number rather than a model — with `max_iters` saying how deep
//! the deepest thread went and `ran_to_end` how many threads left by the loop
//! bound rather than by the early exit.
//!
//! **PRE-REGISTERED, before the run:**
//!
//! - STALE-POLL: the overrun is ≈ 0 on most searches and ≈ 19 × stride on a
//!   minority — and on those, *the same at scan 8 and at scan 64*, because a
//!   stop bounded by time does not care about the bound. `max_iters` reads
//!   `ceil(h/stride) + ~20`, never `count/stride`. ⭐ `ran_to_end` is the
//!   discriminator: near ZERO at scan 64, where 512 iterations are available
//!   and a thread stops after ~19; NONZERO at scan 1, where the cap of 8 bites
//!   first.
//! - NO OVERRUN: `executed − (h + stride)` is ≈ 0 everywhere, including on the
//!   slow searches. Then no thread over-scans, the slow launches run the SAME
//!   permutations more slowly, the whole straggler family is dead, and the cost
//!   is outside this loop and owes a name.
//!
//! # ⛔ The control that decides whether this file may be read at all
//!
//! A counted kernel with different register pressure has different occupancy
//! and therefore measures a different kernel. Every arm therefore runs the
//! SHIPPED kernel on the same seed immediately beside the counted one, and the
//! two must agree on the nonce (always) and on the milliseconds (within the
//! procedure's own noise floor). If the milliseconds disagree, this run reports
//! that the instrument changed the phenomenon and draws no conclusion.
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda \
//!     --test rpx_grind_counted -- --ignored --nocapture
//! ```
//!
//! Needs a GPU. Changes no default and no shipped kernel.
#![cfg(feature = "cuda")]

use std::time::Instant;

use lambda_vm_prover::lfm::algebraic_commit::RpxStarkHash;
use math_cuda::grinding::{GrindCounts, Knobs};
use stark::config::GrindingDigest;
use stark::grinding::{inner_hash_felts, is_valid_nonce};

type RpxGrind = GrindingDigest<RpxStarkHash>;

/// The production grinding factor: this is about the launch, not the bits.
const GRINDING_FACTOR: u8 = 20;

/// The same 256 seeds v5 used, so the two runs describe the same population.
const RUNS: usize = 256;

/// The noise floor v5 measured on this box: the repeated control's median
/// per-seed ratio came back 0.9992. The admissibility control below allows
/// five times that, because it compares two DIFFERENT kernels and a tie is not
/// what is being claimed — only that the twin did not change the phenomenon.
const ADMISSIBLE_MS_RATIO: f64 = 0.05;

/// The arms: the two scan factors that bracket the saturation, the one above
/// it, and the grid that moves the typical seed.
fn arms() -> Vec<(&'static str, Knobs)> {
    vec![
        (
            "scan 1",
            Knobs {
                scan: 1,
                grid: 1024,
            },
        ),
        (
            "scan 8 (record)",
            Knobs {
                scan: 8,
                grid: 1024,
            },
        ),
        (
            "scan 64",
            Knobs {
                scan: 64,
                grid: 1024,
            },
        ),
        (
            "scan 8 grid 4096",
            Knobs {
                scan: 8,
                grid: 4096,
            },
        ),
        (
            "scan 1 grid 4096",
            Knobs {
                scan: 1,
                grid: 4096,
            },
        ),
    ]
}

fn seed_for(i: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&(i as u64).to_le_bytes());
    seed[8] = 0xA5;
    seed
}

/// The shipped kernel, timed — the control arm of every pair.
fn shipped(seed: &[u8; 32], knobs: Knobs) -> (u64, f64) {
    let felts = inner_hash_felts::<RpxGrind>(seed, GRINDING_FACTOR);
    let started = Instant::now();
    let nonce = math_cuda::grinding::generate_nonce_rpx_gpu_at(&felts, GRINDING_FACTOR, knobs)
        .expect("GPU RPX grind (needs a GPU)");
    (nonce, started.elapsed().as_secs_f64() * 1000.0)
}

/// The counted twin, timed the same way.
fn counted(seed: &[u8; 32], knobs: Knobs) -> (GrindCounts, f64) {
    let felts = inner_hash_felts::<RpxGrind>(seed, GRINDING_FACTOR);
    let started = Instant::now();
    let counts = math_cuda::grinding::search_counted(&felts, GRINDING_FACTOR, knobs)
        .expect("counted RPX grind (needs a GPU)");
    (counts, started.elapsed().as_secs_f64() * 1000.0)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing"));
    v[v.len() / 2]
}

#[test]
#[ignore = "device diagnostic; run with --ignored --nocapture on the GPU box"]
fn what_the_slow_grind_launches_actually_execute() {
    let arms = arms();
    let seeds: Vec<[u8; 32]> = (0..RUNS).map(seed_for).collect();

    // Warm-up, excluded by name, on BOTH kernels: the first launch of each
    // pays its own cubin load.
    let w1 = shipped(&seeds[0], Knobs::DEFAULT);
    let w2 = counted(&seeds[0], Knobs::DEFAULT);
    println!(
        "WARM-UP (EXCLUDED): shipped {:.3} ms, counted {:.3} ms, nonces {} / {}",
        w1.1, w2.1, w1.0, w2.0.nonce
    );

    // Per arm: the mean overrun in strides, the summed `ran_to_end`, and
    // whether the twin was admissible. The cross-arm verdict is computed from
    // these HERE rather than by whoever reads the log — the same reason the
    // count slope is printed beside its prediction in `rpx_grind_bench`.
    let mut summary: Vec<(&'static str, f64, u64, bool)> = Vec::new();

    for (name, knobs) in arms.iter() {
        let stride = knobs.stride(math_cuda::grinding::RPX_BLOCK_DIM);
        let block = knobs.block(GRINDING_FACTOR);
        let mut ship_ms = Vec::with_capacity(RUNS);
        let mut cnt_ms = Vec::with_capacity(RUNS);
        let mut rows: Vec<(usize, GrindCounts, f64, f64, i128)> = Vec::with_capacity(RUNS);

        for (s, seed) in seeds.iter().enumerate() {
            // Paired and adjacent, so a clock that drifts drifts through both.
            let (nonce, t_ship) = shipped(seed, *knobs);
            let (counts, t_cnt) = counted(seed, *knobs);

            // ⛔ THE ANSWER IS PINNED FIRST. A diagnostic that returns a
            // different nonce is measuring a different search.
            assert_eq!(
                counts.nonce, nonce,
                "{name}, seed {s}: the counted kernel returned {} and the \
                 shipped kernel {nonce} — they are not running the same search",
                counts.nonce
            );
            assert!(
                is_valid_nonce::<RpxGrind>(seed, counts.nonce, GRINDING_FACTOR),
                "{name}, seed {s}: nonce {} fails is_valid_nonce",
                counts.nonce
            );

            // The model: every nonce below the hit, plus one stride round for
            // the threads that were mid-permutation when it landed. Misses
            // before the hitting block cost their whole block.
            let ideal = (counts.launches - 1) * block + counts.nonce + stride;
            let overrun = counts.executed as i128 - ideal as i128;
            ship_ms.push(t_ship);
            cnt_ms.push(t_cnt);
            rows.push((s, counts, t_ship, t_cnt, overrun));
        }

        // ── the admissibility control, before any reading ──────────────────
        let (m_ship, m_cnt) = (mean(&ship_ms), mean(&cnt_ms));
        let ratio = m_cnt / m_ship;
        let admissible = (ratio - 1.0).abs() <= ADMISSIBLE_MS_RATIO;
        println!(
            "\n=== {name}: count {block}, stride {stride}, {} iterations available ===",
            block / stride
        );
        println!(
            "ADMISSIBILITY: shipped {m_ship:.3} ms vs counted {m_cnt:.3} ms, ratio {ratio:.4} \
             (allowed |1 - r| <= {ADMISSIBLE_MS_RATIO}) => {}",
            if admissible {
                "ADMISSIBLE — the twin did not change the phenomenon"
            } else {
                "NOT ADMISSIBLE — the counters changed the kernel; read no overrun from this arm"
            }
        );

        let overruns: Vec<f64> = rows.iter().map(|r| r.4 as f64).collect();
        let affected = rows.iter().filter(|r| r.4 > stride as i128).count();
        let to_end: u64 = rows.iter().map(|r| r.1.ran_to_end).sum();
        println!(
            "OVERRUN (executed - ideal), permutations: mean {:.0} · median {:.0} · \
             in strides mean {:.2} · searches overrunning by > 1 stride {affected}/{RUNS}",
            mean(&overruns),
            median(&overruns),
            mean(&overruns) / stride as f64,
        );
        println!(
            "RAN_TO_END (threads leaving by the loop bound, summed over {RUNS} searches): \
             {to_end}"
        );
        summary.push((name, mean(&overruns) / stride as f64, to_end, admissible));

        // The ten searches with the largest overrun, with everything that could
        // explain them.
        let mut order: Vec<usize> = (0..RUNS).collect();
        order.sort_by(|a, b| rows[*b].4.cmp(&rows[*a].4));
        println!(
            "{:>5} {:>12} {:>8} {:>14} {:>11} {:>10} {:>9} {:>9}",
            "seed",
            "h",
            "launches",
            "overrun perms",
            "in strides",
            "max_iters",
            "ship ms",
            "cnt ms"
        );
        for &i in order.iter().take(10) {
            let (s, c, t_ship, t_cnt, over) = &rows[i];
            println!(
                "{s:>5} {:>12} {:>8} {over:>14} {:>11.2} {:>10} {t_ship:>9.3} {t_cnt:>9.3}",
                c.nonce,
                c.launches,
                *over as f64 / stride as f64,
                c.max_iters,
            );
        }
    }

    // ── ★ THE VERDICT, by name, beside what each branch predicted ──────────
    let at =
        |n: &str| -> Option<&(&'static str, f64, u64, bool)> { summary.iter().find(|r| r.0 == n) };
    println!("\n=== ★ THE OVERRUN VERDICT ===");
    println!(
        "{:<20} {:>16} {:>14} {:>14}",
        "arm", "overrun/stride", "ran_to_end", "admissible"
    );
    for (name, over, ends, ok) in &summary {
        println!("{name:<20} {over:>16.2} {ends:>14} {:>14}", ok);
    }
    match (at("scan 8 (record)"), at("scan 64")) {
        (Some(s8), Some(s64)) if s8.3 && s64.3 => {
            // Time-bounded means the SAME overrun however much room the loop
            // bound leaves; count-bounded would have grown eightfold here.
            let grew = s64.1 / s8.1.max(1e-9);
            if s8.1 > 5.0 && (0.5..=2.0).contains(&grew) {
                println!(
                    "  ⇒ STALE-POLL: the overrun is {:.1} strides at scan 8 and {:.1} at scan \
                     64, a ratio of {grew:.2} where a loop-bound cause would read about 8. \
                     Threads DO execute extra permutations and the excess is bounded by TIME, \
                     not by `count` — a poll of `*result` served stale. The fix is reader-side.",
                    s8.1, s64.1
                );
            } else if s8.1 <= 1.0 && s64.1 <= 1.0 {
                println!(
                    "  ⇒ NO OVERRUN: {:.2} and {:.2} strides. Nothing over-scans; the slow \
                     launches run the SAME permutations more slowly, the straggler family is \
                     dead, and the cost is outside this loop and owes a name.",
                    s8.1, s64.1
                );
            } else {
                println!(
                    "  ⇒ NEITHER BRANCH: {:.2} strides at scan 8 and {:.2} at scan 64 (ratio \
                     {grew:.2}) match neither pre-registered reading. Report it unresolved \
                     rather than rounding it to a verdict.",
                    s8.1, s64.1
                );
            }
        }
        (Some(_), Some(_)) => println!(
            "  ⛔ NO VERDICT: one of the two arms was NOT ADMISSIBLE, so its counters describe \
             a kernel with different occupancy from the one that ships."
        ),
        _ => {
            println!("  ⛔ NO VERDICT: the two arms the verdict is written over did not both run.")
        }
    }

    println!(
        "\n=== HOW TO READ IT (pre-registered, not re-derived) ===\n  \
         STALE-POLL: the overrun is ~0 on most searches and ~19 strides on a minority, the \
         SAME on those at scan 8 and scan 64; max_iters ~ ceil(h/stride) + 20, never \
         count/stride; ran_to_end near ZERO at scan 64 and NONZERO at scan 1.\n  \
         NO OVERRUN: the overrun is ~0 everywhere including the slow searches — nothing \
         over-scans, the slow launches run the same permutations more slowly, and the cost \
         is outside this loop.\n  \
         ⚠ An arm reported NOT ADMISSIBLE says nothing either way: its counters describe a \
         kernel with different occupancy from the one that ships."
    );
}
