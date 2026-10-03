//! ★ ROUND 3: is the grind's stale-poll overrun CONTENTION on one address?
//!
//! # What is being decided
//!
//! The grind is the single largest remaining stage on the WHIR block. O3's
//! stage B established the shape of the cost: threads keep scanning for a
//! bounded TIME after the answer is known (the excess saturates above `count`
//! rather than growing with it, and fits `T·(1 − e^(−(N−8)/τ))` with τ ≈ 19), a
//! poll of `*result` served stale — not threads running to the loop bound. O4
//! then read the SASS: the poll is already `LDG.E.64.STRONG.SYS` inside the
//! loop, the strongest system-scope load, so a cache qualifier cannot help
//! (`__ldcg`/`__ldcv` are WEAKER and dead). The one surviving explanation is
//! CONTENTION: 131,072 resident threads each issue that system-scope 8-byte load
//! to ONE address every permutation, and the finder's `atomicMin` queues behind
//! the flood, so the answer takes longer to land and to propagate — and while it
//! is stale, threads over-scan.
//!
//! Contention cannot be tested by raising the grid (grid 1024 already holds
//! 131,072 of the card's 174,080 resident threads). It CAN be tested by lowering
//! the poll RATE. `rpx_grind_search_counted` gained a `poll_period` launch
//! parameter: it polls `*result` only every `poll_period`-th iteration,
//! STAGGERED by thread so at any one iteration only `1/poll_period` of the
//! threads issue the load — the average AND the peak request rate both fall by
//! `poll_period`, with the same residency, the same loop bound, the same nonce
//! order. Only the flood on that address changes.
//!
//! # ★ THE SIGN, PRE-REGISTERED (before the run; judged against it after)
//!
//! The measured quantity is the OVERRUN — `executed − ideal`, `ideal =
//! nonce + stride` — in strides (≈ mean extra iterations per thread), at the
//! record posture scan 8 / grid 1024, as `poll_period` (`k`) rises 1 → 4 → 16 →
//! 64:
//!
//! - overrun FALLS as k rises ⇒ CONTENTION. The flood is the cause; the fix is a
//!   warp-lane poll + `__shfl_sync` broadcast (32× fewer requests) and/or one
//!   thread per block polling a `__shared__` cell holding the VALUE (not a flag).
//! - overrun RISES ≈ linearly in k ⇒ the poll is merely coarser (a thread that
//!   polls every k-th iteration over-scans up to k−1 extra iterations after the
//!   answer lands, ≈ (k−1)/2 on average). The whole "poll less" family dies
//!   together and the only survivor is the LDL.64 sponge lever.
//! - FLAT ⇒ neither; a different discriminator is owed.
//!
//! A SIGN, not a magnitude — it survives a mispredicted size, which O3's τ ≈ 19
//! did not. The `(k−1)/2` granularity term is exactly the "rises ≈ linearly"
//! prediction; the `adjusted = overrun − (k−1)/2` column removes it to isolate
//! the contention component, which matters most at k=64 where granularity alone
//! is +31.5 strides and can dominate. k=4 and k=16 are the primary reads.
//!
//! # The controls, both mandatory
//!
//! - ADMISSIBILITY at k=1: the counted twin at `poll_period = 1` polls every
//!   iteration, the shipped kernel's rate, so its milliseconds must reproduce
//!   the shipped kernel's within the noise floor. If they do not, the counters
//!   changed the kernel's occupancy and NO sign may be read from the sweep. At
//!   k>1 the twin is DELIBERATELY different (that is the manipulation), so its
//!   ms are reported as a POLL EFFECT, never gated.
//! - SOUNDNESS: the winning nonce is IDENTICAL across every k, for every seed.
//!   The poll can only make a thread STOP scanning early; `*result` holds
//!   `U64_MAX` or a nonce that passed `digest[0] < limit`; a thread's nonces
//!   only increase; `atomicMin` only ever LOWERS `*result`, so a staler value is
//!   a LARGER one and polling less can only DELAY an exit, never accept a wrong
//!   nonce or change the one a launch returns. This control is what witnesses
//!   that; if any k changes a winning nonce, the argument is wrong and we stop.
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda \
//!     --test rpx_grind_counted -- --ignored --nocapture
//! ```
//!
//! Needs a GPU. Changes no default and no shipped kernel — the `poll_period`
//! parameter lives only on the diagnostic twin, which is on no proving path.
#![cfg(feature = "cuda")]

use std::time::Instant;

use lambda_vm_prover::lfm::algebraic_commit::RpxStarkHash;
use math_cuda::grinding::{GrindCounts, Knobs};
use stark::config::GrindingDigest;
use stark::grinding::{inner_hash_felts, is_valid_nonce};

type RpxGrind = GrindingDigest<RpxStarkHash>;

/// The production grinding factor: this is about the launch, not the bits.
const GRINDING_FACTOR: u8 = 20;

/// The same 256 seeds O3 used, so this run describes the same population.
const RUNS: usize = 256;

/// The noise floor O3 measured on this box: the repeated control's median
/// per-seed ratio came back 0.9992. The admissibility control allows five times
/// that, because it compares two DIFFERENT kernels (counted vs shipped) and only
/// claims the twin did not change the phenomenon, not that they tie.
const ADMISSIBLE_MS_RATIO: f64 = 0.05;

/// The margin, in strides, a mean overrun must move across the sweep to be read
/// as a sign rather than run-to-run noise. This is a SIGN read; the authoritative
/// call is made by hand from the printed table, this only keeps the automated
/// verdict from flapping on noise.
const SIGN_MARGIN_STRIDES: f64 = 1.0;

/// The four arms: the SAME record posture (scan 8, grid 1024) at four poll
/// periods. Only the poll rate changes — that is the whole manipulation.
fn arms() -> Vec<(&'static str, Knobs, u64)> {
    let posture = Knobs {
        scan: 8,
        grid: 1024,
    };
    vec![
        ("k=1 (shipped rate)", posture, 1),
        ("k=4", posture, 4),
        ("k=16", posture, 16),
        ("k=64", posture, 64),
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

/// The counted twin at a given poll period, timed the same way.
fn counted(seed: &[u8; 32], knobs: Knobs, poll_period: u64) -> (GrindCounts, f64) {
    let felts = inner_hash_felts::<RpxGrind>(seed, GRINDING_FACTOR);
    let started = Instant::now();
    let counts = math_cuda::grinding::search_counted(&felts, GRINDING_FACTOR, knobs, poll_period)
        .expect("counted RPX grind (needs a GPU)");
    (counts, started.elapsed().as_secs_f64() * 1000.0)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// One arm's aggregate read, computed once and used by both the per-arm print
/// and the cross-arm verdict — so the verdict is not recomputed by whoever reads
/// the log (the failure mode that produced a verdict which could not pass).
struct ArmRead {
    name: &'static str,
    poll_period: u64,
    /// mean overrun over the seeds, in strides ≈ mean extra iterations/thread.
    overrun_strides: f64,
    /// `overrun_strides − (k−1)/2`: the granularity term removed, leaving the
    /// stale-window (contention) component.
    adjusted: f64,
    ran_to_end: u64,
    max_iters: u64,
    /// `Some` only for the k=1 arm; `None` for the k>1 arms, which are meant to
    /// differ from the shipped kernel.
    admissible: Option<bool>,
}

#[test]
#[ignore = "device diagnostic; run with --ignored --nocapture on the GPU box"]
fn what_lowering_the_grind_poll_rate_does_to_the_overrun() {
    let arms = arms();
    let n_arms = arms.len();
    let seeds: Vec<[u8; 32]> = (0..RUNS).map(seed_for).collect();
    let block_dim = math_cuda::grinding::RPX_BLOCK_DIM;

    // Warm-up, excluded by name, on BOTH kernels: the first launch of each pays
    // its own cubin load.
    let w1 = shipped(&seeds[0], Knobs::DEFAULT);
    let w2 = counted(&seeds[0], Knobs::DEFAULT, 1);
    println!(
        "WARM-UP (EXCLUDED): shipped {:.3} ms, counted {:.3} ms, nonces {} / {}",
        w1.1, w2.1, w1.0, w2.0.nonce
    );

    // Per-arm accumulators, indexed by the canonical arm order.
    let mut ship_ms: Vec<Vec<f64>> = vec![Vec::with_capacity(RUNS); n_arms];
    let mut cnt_ms: Vec<Vec<f64>> = vec![Vec::with_capacity(RUNS); n_arms];
    let mut overruns: Vec<Vec<f64>> = vec![Vec::with_capacity(RUNS); n_arms]; // permutations
    let mut to_end_sum: Vec<u64> = vec![0; n_arms];
    let mut max_iters_max: Vec<u64> = vec![0; n_arms];
    let mut affected: Vec<usize> = vec![0; n_arms];
    // The winning nonce per seed at each arm — the cross-k soundness control.
    let mut nonces: Vec<Vec<u64>> = vec![Vec::with_capacity(RUNS); n_arms];
    // The poll_period the launch REPORTS back, so the banner quotes the knob as
    // it reached the search, not as the caller believes it passed it.
    let mut poll_seen: Vec<u64> = vec![0; n_arms];

    for (s, seed) in seeds.iter().enumerate() {
        // Rotate the arm order per seed so any drift over the run spreads across
        // arms rather than loading onto the last one, and pair shipped-vs-counted
        // adjacently so a clock that drifts drifts through both.
        for j in 0..n_arms {
            let a = (j + s) % n_arms;
            let (name, knobs, poll_period) = arms[a];
            let stride = knobs.stride(block_dim);

            let (nonce, t_ship) = shipped(seed, knobs);
            let (counts, t_cnt) = counted(seed, knobs, poll_period);

            // ⛔ THE ANSWER IS PINNED FIRST. A diagnostic that returns a
            // different nonce is measuring a different search.
            assert_eq!(
                counts.nonce, nonce,
                "{name}, seed {s}: the counted kernel returned {} and the shipped \
                 kernel {nonce} — they are not running the same search",
                counts.nonce
            );
            assert!(
                is_valid_nonce::<RpxGrind>(seed, counts.nonce, GRINDING_FACTOR),
                "{name}, seed {s}: nonce {} fails is_valid_nonce",
                counts.nonce
            );
            // The knob must reach the launch: the twin echoes back the
            // poll_period it was launched with.
            assert_eq!(
                counts.poll_period, poll_period,
                "{name}, seed {s}: the launch used poll_period {} not {poll_period}",
                counts.poll_period
            );

            // ideal = every nonce below the hit, plus one stride round for the
            // threads mid-permutation when it landed. `nonce` is absolute, so
            // any missed blocks are already counted in it (O3's double-count
            // trap): it is `nonce + stride`, full stop. Every seed here hits on
            // its first launch at scan 8.
            let ideal = counts.nonce + stride;
            let overrun = counts.executed as i128 - ideal as i128;

            ship_ms[a].push(t_ship);
            cnt_ms[a].push(t_cnt);
            overruns[a].push(overrun as f64);
            to_end_sum[a] += counts.ran_to_end;
            if counts.max_iters > max_iters_max[a] {
                max_iters_max[a] = counts.max_iters;
            }
            if overrun > stride as i128 {
                affected[a] += 1;
            }
            nonces[a].push(counts.nonce);
            poll_seen[a] = counts.poll_period;
        }
    }

    // ── the KNOBS banner: the poll period varied and reached the search ──────
    for (a, (name, knobs, poll_period)) in arms.iter().enumerate() {
        println!(
            "★ GRIND KNOBS: {name} — scan {} grid {} poll_period {} (launch reported {})",
            knobs.scan, knobs.grid, poll_period, poll_seen[a]
        );
        assert_eq!(
            poll_seen[a], *poll_period,
            "{name}: the launch reported poll_period {} not {poll_period}",
            poll_seen[a]
        );
    }

    // ── the cross-k IDENTICAL-NONCE soundness control ───────────────────────
    //
    // ⛔ THE BOUNDS ARE THE CHECK, so they are spelled out rather than trusted
    // to an iterator that looks equivalent. `nonces` is
    // `vec![Vec::with_capacity(RUNS); n_arms]` and the seed loop pushes exactly
    // once per (seed, arm) with no early exit, so `nonces.len() == n_arms` and
    // every row is `RUNS` long: iterating `nonces[0]` is the old `0..RUNS`, and
    // `take(n_arms).skip(1)` is the old `1..n_arms`.
    //
    // ⚠ CLIPPY'S OWN SUGGESTION HERE IS WRONG. `needless_range_loop` offers
    // `for <item> in nonces.iter().take(n_arms).skip(1)` for the OUTER loop too,
    // which drops the `s` index — and the body needs both indices, because the
    // comparison is `nonces[a][s]` against `nonces[0][s]`. Taking that
    // suggestion would have compared whole arms instead of per-seed nonces and
    // turned a control that can fail into one that cannot.
    let mut nonce_mismatch = 0usize;
    for (s, n0) in nonces[0].iter().enumerate() {
        for arm in nonces.iter().take(n_arms).skip(1) {
            if arm[s] != *n0 {
                nonce_mismatch += 1;
            }
        }
    }
    println!(
        "\nSOUNDNESS (identical winning nonce across all k, per seed): {nonce_mismatch} \
         mismatch(es) over {RUNS} seeds"
    );
    assert_eq!(
        nonce_mismatch, 0,
        "a poll period changed a winning nonce — the soundness argument is wrong, stop"
    );

    // ── per-arm read + admissibility (k=1) / poll effect (k>1) ──────────────
    let m_cnt_k1 = mean(&cnt_ms[0]); // arm 0 is k=1 by construction
    let mut summary: Vec<ArmRead> = Vec::with_capacity(n_arms);
    for a in 0..n_arms {
        let (name, knobs, poll_period) = arms[a];
        let stride = knobs.stride(block_dim) as f64;
        let block = knobs.block(GRINDING_FACTOR);
        let o_perms_mean = mean(&overruns[a]);
        let o_strides = o_perms_mean / stride;
        let granularity = (poll_period as f64 - 1.0) / 2.0;
        let adjusted = o_strides - granularity;

        println!(
            "\n=== {name}: poll_period {poll_period}, count {block}, stride {}, {} iterations available ===",
            knobs.stride(block_dim),
            block / knobs.stride(block_dim)
        );

        let m_ship = mean(&ship_ms[a]);
        let m_cnt = mean(&cnt_ms[a]);
        let admissible = if poll_period == 1 {
            let ratio = m_cnt / m_ship;
            let ok = (ratio - 1.0).abs() <= ADMISSIBLE_MS_RATIO;
            println!(
                "ADMISSIBILITY: shipped {m_ship:.3} ms vs counted {m_cnt:.3} ms, ratio {ratio:.4} \
                 (allowed |1 - r| <= {ADMISSIBLE_MS_RATIO}) => {}",
                if ok {
                    "ADMISSIBLE — the twin did not change the phenomenon"
                } else {
                    "NOT ADMISSIBLE — the counters changed the kernel; read no sign from this sweep"
                }
            );
            Some(ok)
        } else {
            let ratio = m_cnt / m_cnt_k1;
            println!(
                "POLL EFFECT: counted(k={poll_period}) {m_cnt:.3} ms vs counted(k=1) \
                 {m_cnt_k1:.3} ms, ms ratio {ratio:.4} (informational; the sign is read from the \
                 overrun below, not the wall clock)"
            );
            None
        };

        println!(
            "OVERRUN (executed - ideal): mean {o_perms_mean:.0} perms · in strides {o_strides:.2} · \
             granularity (k-1)/2 = {granularity:.1} · adjusted (overrun - granularity) {adjusted:.2} \
             · searches overrunning > 1 stride {}/{RUNS}",
            affected[a]
        );
        println!(
            "RAN_TO_END (threads leaving by the loop bound, summed over {RUNS}): {} · deepest \
             thread (max_iters) {}",
            to_end_sum[a], max_iters_max[a]
        );

        summary.push(ArmRead {
            name,
            poll_period,
            overrun_strides: o_strides,
            adjusted,
            ran_to_end: to_end_sum[a],
            max_iters: max_iters_max[a],
            admissible,
        });
    }

    // ── ★ THE VERDICT, by name, beside what each branch predicted ───────────
    println!("\n=== ★ THE OVERRUN VERDICT ===");
    println!(
        "{:<20} {:>12} {:>16} {:>12} {:>12} {:>12} {:>10}",
        "arm",
        "poll_period",
        "overrun/stride",
        "granularity",
        "adjusted",
        "ran_to_end",
        "max_iters"
    );
    for r in &summary {
        let gran = (r.poll_period as f64 - 1.0) / 2.0;
        println!(
            "{:<20} {:>12} {:>16.2} {:>12.1} {:>12.2} {:>12} {:>10}",
            r.name, r.poll_period, r.overrun_strides, gran, r.adjusted, r.ran_to_end, r.max_iters
        );
    }

    let at = |pp: u64| -> Option<&ArmRead> { summary.iter().find(|r| r.poll_period == pp) };
    let adm_k1 = at(1).and_then(|r| r.admissible).unwrap_or(false);
    match (adm_k1, at(1), at(16)) {
        (false, _, _) => println!(
            "  ⛔ NO VERDICT: the k=1 arm was NOT ADMISSIBLE (or absent); its counters describe a \
             kernel with different occupancy from the one that ships, so the sweep's baseline is void."
        ),
        (true, Some(r1), Some(r16)) => {
            let (o1, o16) = (r1.overrun_strides, r16.overrun_strides);
            let adj16 = r16.adjusted;
            if o16 < o1 - SIGN_MARGIN_STRIDES {
                println!(
                    "  ⇒ CONTENTION: overrun FELL {o1:.2} -> {o16:.2} strides (k=1 -> k=16) as the \
                     poll rate dropped 16x. The stale-poll overrun is driven by contention on the \
                     one *result address; a warp-lane (__shfl_sync) or per-block (__shared__ cell \
                     holding the VALUE) poll is the fix. Granularity-adjusted k=16 = {adj16:.2} vs \
                     k=1 {o1:.2}."
                );
            } else if o16 > o1 + SIGN_MARGIN_STRIDES {
                println!(
                    "  ⇒ POLL FAMILY DEAD: overrun ROSE {o1:.2} -> {o16:.2} strides (k=1 -> k=16); \
                     polling less only wastes more scanning. Granularity alone predicts +{:.1}; \
                     granularity-adjusted k=16 = {adj16:.2} vs k=1 {o1:.2} (≈ equal ⇒ the rise is \
                     pure granularity). The LDL.64 sponge lever is the only survivor.",
                    (16.0 - 1.0) / 2.0
                );
            } else {
                println!(
                    "  ⇒ FLAT: overrun barely moved {o1:.2} -> {o16:.2} strides at 16x fewer polls; \
                     neither contention nor a granularity-linear rise. A different discriminator is \
                     owed."
                );
            }
        }
        _ => println!(
            "  ⛔ NO VERDICT: the k=1 and k=16 arms the verdict is written over did not both run."
        ),
    }

    // How to read it, pre-registered — worded to AVOID the verdict's own branch
    // markers, so a launcher that counts those markers is not fooled by its own
    // explanation (O3's stage-B miscount).
    println!(
        "\n=== HOW TO READ IT (pre-registered, not re-derived) ===\n  \
         Overrun FALLING as the poll period rises => the flood on *result is the cause; the \
         poll-less fix (warp-lane or per-block, the cell holding the VALUE) lives.\n  \
         Overrun RISING ~linearly in the poll period => the poll is merely coarser (each thread \
         over-scans ~(k-1)/2 iterations after the answer lands); that whole fix family is closed \
         and only the LDL.64 sponge lever remains.\n  \
         No movement => a different discriminator is owed.\n  \
         k=64 is granularity-dominant ((k-1)/2 = 31.5 strides), so it confirms the trend; k=4 and \
         k=16 are the primary reads. The 'adjusted' column removes the (k-1)/2 granularity term to \
         isolate the contention component at every k."
    );
}
