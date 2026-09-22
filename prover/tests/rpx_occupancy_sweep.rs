//! ★ ROUND 3 DISCRIMINATOR: is the RPX permutation latency-bound or compute-bound?
//!
//! # Why this exists
//!
//! Round 3's poll k-sweep falsified the contention hypothesis (POLL FAMILY DEAD:
//! the overrun rose purely with poll granularity, zero contention component). So
//! the grind's cost is the necessary scanning, and the three big stages — grind
//! (14.1s) + two hashing passes (15.1+16.4s) = 45.6s — are all bound by the RPX
//! PERMUTATION. The file's own cost model (rpx.cu:40-61) says the permutation is
//! compute-heavy: the inverse S-box (x^(1/7)) is 2592/2736 = 95% of the field
//! multiplications, a serial 72-deep chain × 12 lanes, and the phase-2 note
//! points tuning at REGISTER PRESSURE.
//!
//! The card runs the RPX kernels at ~2/3 occupancy, register-limited (~64
//! regs/thread ⇒ 8 blocks/SM). The open question that decides round 3's lever:
//! does RAISING occupancy (fewer registers per thread, more blocks per SM) speed
//! the permutation? This sweep answers it by capping registers at build time
//! (`-maxrregcount`, via `LAMBDA_VM_RPX_MAXRREGCOUNT`) and timing the permutation
//! at each cap. It uses `-maxrregcount` rather than `__launch_bounds__` because
//! the SHIPPED grind kernel — which the sweep must move — cannot carry a
//! per-kernel annotation (it is off-limits), and launch-bounds variants would
//! need new kernels loaded in `device.rs` (also off-limits). A whole-cubin
//! register cap moves the untouched production kernels without editing them.
//!
//! # ★ THE SIGN, PRE-REGISTERED (read ACROSS the builds, not within one run)
//!
//! One build is one register cap is one point. Run this once per cap (unset/255
//! = baseline ~64 regs; 85 → ~6 blocks/SM; 51 → ~10; 42 → ~12 = full), then read
//! the permutation time (probe ns/permutation, and the grind mean ms) against the
//! ACHIEVED blocks/SM:
//!
//! - time FALLS as occupancy rises ⇒ LATENCY-bound ⇒ round 3's lever is (2),
//!   reduce the permutation's register pressure so more blocks fit (sizing
//!   ~45.6 → 35-41s if the fall is proportional). Build that lever next.
//! - time RISES or is FLAT as occupancy rises (the register cap spills the
//!   inverse-S-box working set to local memory, or the SM is already
//!   issue-saturated) ⇒ COMPUTE/issue-bound ⇒ levers (1) LDL.64 and (2)
//!   occupancy are both dead, and only (3) a KAT-preserving arithmetic speedup of
//!   the inverse S-box remains (small, delicate) — or round 3 lands near the
//!   permutation's arithmetic floor, an honest close.
//!
//! A SIGN, not a magnitude. A register cap cannot change a KERNEL RESULT (it only
//! moves spills and occupancy), so there is no soundness question here; the
//! permutation's device parity is pinned separately by `rpx_device_parity`.
//!
//! ```text
//! LAMBDA_VM_RPX_MAXRREGCOUNT=<cap> \
//!   cargo test -p lambda-vm-prover --release --features cuda \
//!     --test rpx_occupancy_sweep -- --ignored --nocapture
//! ```
//!
//! Needs a GPU, and a REBUILD per cap (the cubin is recompiled when the env
//! changes). Changes no default and no shipped kernel.
#![cfg(feature = "cuda")]

use std::time::Instant;

use lambda_vm_prover::lfm::algebraic_commit::RpxStarkHash;
use math_cuda::grinding::Knobs;
use stark::config::GrindingDigest;
use stark::grinding::inner_hash_felts;

type RpxGrind = GrindingDigest<RpxStarkHash>;

/// The production grinding factor; this is about the launch, not the bits.
const GRINDING_FACTOR: u8 = 20;

/// Pure-permutation probe size: states per launch × timed launches. ~268M
/// permutations, enough for a stable time while the one host-to-device copy is
/// amortised away.
const PROBE_N: usize = 1 << 20;
const PROBE_ITERS: usize = 256;

/// Grind seeds averaged for the real-stage confirm.
const GRIND_SEEDS: usize = 64;

fn seed_for(i: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&(i as u64).to_le_bytes());
    seed[8] = 0xA5;
    seed
}

/// One grind, timed — the real stage the permutation dominates.
fn grind_once(seed: &[u8; 32], knobs: Knobs) -> f64 {
    let felts = inner_hash_felts::<RpxGrind>(seed, GRINDING_FACTOR);
    let started = Instant::now();
    let _ = math_cuda::grinding::generate_nonce_rpx_gpu_at(&felts, GRINDING_FACTOR, knobs)
        .expect("GPU RPX grind (needs a GPU)");
    started.elapsed().as_secs_f64() * 1000.0
}

#[test]
#[ignore = "device diagnostic; run once per LAMBDA_VM_RPX_MAXRREGCOUNT build on the GPU box"]
fn how_occupancy_moves_the_rpx_permutation() {
    let cap = std::env::var("LAMBDA_VM_RPX_MAXRREGCOUNT").unwrap_or_else(|_| "unset".to_string());

    // ── the pure permutation: throughput at the achieved register cap ────────
    let probe = math_cuda::rpx::permute_probe_sweep(PROBE_N, PROBE_ITERS)
        .expect("permute probe sweep (needs a GPU)");
    let total_perms = probe.n * probe.iters;
    let ns_per_perm = probe.secs * 1e9 / total_perms as f64;

    // ── the real stage: the grind, mean over seeds (warm-up excluded) ────────
    let knobs = Knobs {
        scan: 8,
        grid: 1024,
    };
    let _ = grind_once(&seed_for(0), knobs);
    let mut grind_ms = Vec::with_capacity(GRIND_SEEDS);
    for i in 0..GRIND_SEEDS {
        grind_ms.push(grind_once(&seed_for(i), knobs));
    }
    let grind_mean = grind_ms.iter().sum::<f64>() / grind_ms.len() as f64;
    let (grind_regs, grind_blocks) =
        math_cuda::grinding::grind_occupancy().expect("grind occupancy (needs a GPU)");

    // ── the report: the KNOBS banner proves the cap reached the cubin, then one
    //    machine-readable line the launcher lifts and collects across builds ──
    println!(
        "★ OCC KNOBS: maxrregcount={cap} · probe regs {} blocks/SM {} · grind regs {grind_regs} \
         blocks/SM {grind_blocks}",
        probe.regs, probe.blocks_per_sm
    );
    println!(
        "PROBE: {} states x {} iters = {total_perms} permutations in {:.4} s => \
         {ns_per_perm:.3} ns/permutation",
        probe.n, probe.iters, probe.secs
    );
    println!("GRIND: mean {grind_mean:.3} ms over {GRIND_SEEDS} seeds (scan 8 grid 1024)");
    println!(
        "OCCSWEEP maxrregcount={cap} probe_regs={} probe_blocks_per_sm={} \
         probe_ns_per_perm={ns_per_perm:.3} grind_regs={grind_regs} \
         grind_blocks_per_sm={grind_blocks} grind_mean_ms={grind_mean:.3}",
        probe.regs, probe.blocks_per_sm
    );
}
