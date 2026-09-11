//! ★ The D′a gate: is there vector headroom in the Goldilocks multiply on this
//! box, before anyone rewrites the RPX permutation?
//!
//! # The question, and why it is worth one file
//!
//! Lane E measured the LFM executor at **83% software RPX permutation** and the
//! permutation at **2,356 ns on Zen 5 / 3,000 on Zen 4**. Lane ② then closed
//! the obvious lever: batching the inverse S-box across permutations is SLOWER
//! at every width above twelve on both x86 parts, so there is no scalar
//! instruction-level parallelism left.
//!
//! What ② did **not** close is the other direction — running the same twelve
//! lanes on WIDER registers. The inverse S-box is ~72 chained multiplications
//! per lane over twelve independent lanes, so its cost is the Goldilocks
//! multiply and nothing else, and the whole question reduces to: **can eight
//! Goldilocks multiplies in a vector beat eight done one at a time?**
//!
//! # What this measures, and what it does NOT
//!
//! One kernel body, written in portable Rust, compiled twice:
//!
//! * plain — the correctness reference, and the scalar-per-lane baseline;
//! * inside `#[target_feature(enable = "avx512f,avx512dq")]` on `x86_64`, which
//!   lets the backend vectorise the same eight lanes onto `zmm` registers.
//!
//! ⚠ **`#[target_feature]` deliberately, and NOT `RUSTFLAGS`.** The repo's build
//! convention forbids ad-hoc flags such as `-C target-cpu=native`: any flag
//! change forks the `sccache` key and every worktree recompiles. A function
//! attribute forks nothing, and the runtime `is_x86_feature_detected!` guard is
//! what keeps the binary portable.
//!
//! ⛔ **This is a LOWER bound on D′a's prize, and the reading must respect that.**
//! It measures what the compiler's auto-vectoriser achieves, not what a
//! hand-written `vpmuludq` kernel with explicit carry handling would. So:
//!
//! | reading | conclusion |
//! |---|---|
//! | **≥ 1.5×** | D′a is OPEN — the headroom exists, an intrinsics kernel is worth writing |
//! | **< 1.5×** | D′a is NOT reachable by auto-vectorisation — which is *not* the same as closed; the next step would be a ~30-line intrinsics spike, on an x86 box, not a green light |
//!
//! Run it with
//! `cargo test --release -p lambda-vm-prover --lib goldilocks_simd -- --ignored --nocapture`.

use crate::tables::types::{FE, GoldilocksField};
use math::field::traits::IsPrimeField;

/// `2^64 − p`, the correction a wrap-around costs. `p = 2^64 − 2^32 + 1`, so
/// this is `2^32 − 1`.
const EPSILON: u64 = 0xFFFF_FFFF;

/// Lanes per batch — eight `u64` is exactly one 512-bit register, and twelve
/// (the state width) is not, so the kernel is written at the register's width
/// and a permutation would run it twice with a four-lane tail.
const LANES: usize = 8;

/// One Goldilocks multiply, branchless.
///
/// `a·b = hi·2^64 + lo`; with `hi = n_hi·2^32 + n_lo` and the field's two
/// identities `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1`,
/// `a·b ≡ lo − n_hi + n_lo·(2^32 − 1)  (mod p)`.
///
/// ⚠ **Branchless on purpose.** The `if borrow { … }` spelling is the readable
/// one and is what kills vectorisation: a data-dependent branch per lane forces
/// the backend back to scalar, which would make this file measure its own
/// mistake rather than the machine. `bool as u64` is a `setcc`, and a vector
/// compare-and-mask in the widened form.
///
/// The result is reduced modulo `p` but **not canonicalised** — it may land in
/// `[p, 2^64)`, exactly like the field crate's own representation, which is why
/// every comparison below goes through [`GoldilocksField::canonical`].
#[inline(always)]
fn mul_reduce(a: u64, b: u64) -> u64 {
    let x = (a as u128) * (b as u128);
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let n_hi = x_hi >> 32;
    let n_lo = x_hi & 0xFFFF_FFFF;

    let (r, borrow) = x_lo.overflowing_sub(n_hi);
    let r = r.wrapping_sub(EPSILON * u64::from(borrow));
    // `n_lo < 2^32`, so `n_lo·(2^32 − 1) = (n_lo << 32) − n_lo` cannot overflow.
    let t = (n_lo << 32) - n_lo;
    let (r, carry) = r.overflowing_add(t);
    r.wrapping_add(EPSILON * u64::from(carry))
}

/// The kernel under test: eight multiplies, accumulated into the state so the
/// optimiser cannot hoist the loop and so successive iterations DEPEND on each
/// other — the same chained shape the inverse S-box has.
#[inline(always)]
fn mul_batch(state: &mut [u64; LANES], k: &[u64; LANES]) {
    for i in 0..LANES {
        state[i] = mul_reduce(state[i], k[i]);
    }
}

/// [`mul_batch`] with AVX-512 enabled for this function only.
///
/// # Safety
///
/// The caller must have checked `is_x86_feature_detected!("avx512f")` and
/// `("avx512dq")`. The body is ordinary safe Rust; the `unsafe` is the
/// attribute's contract, not the arithmetic's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn mul_batch_avx512(state: &mut [u64; LANES], k: &[u64; LANES]) {
    mul_batch(state, k);
}

/// Nanoseconds per Goldilocks multiply, over `rounds · LANES` of them.
///
/// The minimum of `repeats` is reported: this lane's own `rpo::throughput`
/// produced 4,712–6,576 ns for identical code on one machine, and noise only
/// ever adds time, so the fastest run is the closest to the machine.
fn time(label: &str, rounds: usize, repeats: usize, mut run: impl FnMut(&mut [u64; LANES])) -> f64 {
    use std::time::Instant;
    let mut best = f64::MAX;
    let mut guard = 0u64;
    for _ in 0..repeats {
        let mut state: [u64; LANES] = core::array::from_fn(|i| 0x9E37_79B9_7F4A_7C15 ^ i as u64);
        let start = Instant::now();
        run(&mut state);
        let ns = start.elapsed().as_nanos() as f64 / (rounds * LANES) as f64;
        // Consumed after the timer so the loop cannot be optimised away, and
        // asserted below so the consumption itself cannot be.
        guard ^= state.iter().fold(0u64, |a, b| a ^ b);
        best = best.min(ns);
    }
    assert_ne!(guard, 0, "the kernel must produce a value, not be elided");
    println!("  {label:<34} {best:>7.3} ns/mul");
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(x: u64) -> u64 {
        GoldilocksField::canonical(&x)
    }

    /// ★★ The gate on the KERNEL, not on the timing: `mul_reduce` must be the
    /// field's multiplication.
    ///
    /// Runs everywhere, including this lane's aarch64 laptop, because the VALUES
    /// are architecture-independent even though the speed is not. It is what
    /// makes the benchmark below unable to report a number from a wrong kernel —
    /// the failure mode that would otherwise open or close D′a on a lie.
    ///
    /// The corners are deliberate: `p − 1` squared, `2^32` (where the folding
    /// identity switches limbs), and values whose product's high word forces
    /// both the borrow and the carry path.
    #[test]
    fn the_kernel_is_the_field_multiplication() {
        const P: u64 = 0xFFFF_FFFF_0000_0001;
        let mut corners = vec![
            0,
            1,
            2,
            EPSILON,
            1 << 32,
            (1 << 32) + 1,
            u32::MAX as u64,
            P - 1,
            P - 2,
            u64::MAX,
            u64::MAX - 1,
        ];
        // A deterministic spread, so a regression is reproducible.
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..256 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            corners.push(seed);
        }

        for &a in &corners {
            for &b in &corners {
                let got = canon(mul_reduce(a, b));
                let want = (FE::from(a) * FE::from(b)).canonical_u64();
                assert_eq!(
                    got, want,
                    "mul_reduce({a:#x}, {b:#x}) = {got:#x}, field says {want:#x}"
                );
            }
        }
    }

    /// The batch kernel is the scalar one, eight at a time — so the timed
    /// comparison is between two spellings of one function and not between two
    /// functions.
    #[test]
    fn the_batch_is_the_scalar_kernel_eight_at_a_time() {
        let mut state: [u64; LANES] = core::array::from_fn(|i| 0x1234_5678 + i as u64);
        let k: [u64; LANES] = core::array::from_fn(|i| 0xDEAD_BEEF - i as u64);
        let want: [u64; LANES] = core::array::from_fn(|i| mul_reduce(state[i], k[i]));
        mul_batch(&mut state, &k);
        assert_eq!(state, want);
    }

    /// ★ The D′a reading. `#[ignore]`d because it is a timing measurement, not
    /// a property.
    ///
    /// Both arms run the SAME chained kernel over the same lane count; the only
    /// difference is whether the backend was allowed AVX-512 for it. The ratio
    /// is the load-bearing output and the absolutes are not, exactly as
    /// `rpx::ladder` records for the hash candidates.
    #[test]
    #[ignore]
    fn goldilocks_simd_headroom() {
        const ROUNDS: usize = 4_000_000;
        const REPEATS: usize = 5;

        println!("Goldilocks multiply, 8 lanes, chained — this machine:");
        let k: [u64; LANES] = core::array::from_fn(|i| 0xC0FF_EE00_0000_0001 + i as u64);

        let base = time("baseline (no target_feature)", ROUNDS, REPEATS, |s| {
            for _ in 0..ROUNDS {
                mul_batch(s, &k);
            }
        });

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
            {
                let simd = time("avx512f,avx512dq", ROUNDS, REPEATS, |s| {
                    for _ in 0..ROUNDS {
                        // SAFETY: both features were just detected.
                        unsafe { mul_batch_avx512(s, &k) };
                    }
                });
                let ratio = base / simd;
                println!();
                println!("  ⇒ AVX-512 is {ratio:.2}× the baseline");
                println!(
                    "  ⇒ D′a is {}",
                    if ratio >= 1.5 {
                        "OPEN — the headroom exists; an intrinsics kernel is worth writing"
                    } else {
                        "NOT reachable by auto-vectorisation (≠ closed — an intrinsics \
                         spike is the next step, not a green light)"
                    }
                );
                // What the prize would be, if taken, at lane E's measured shares:
                // the permutation is ~83% of `execute`, and `execute` is 1.82 s
                // of a serial wrap.
                println!(
                    "  ⇒ at 83% of a 1.82 s wrap executor: {:.2} s → {:.2} s",
                    1.82,
                    1.82 * (0.17 + 0.83 / ratio.max(1.0)),
                );
            } else {
                println!("  (no AVX-512 on this machine — the SIMD arm did not run)");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = base;
            println!("  (not x86_64 — the SIMD arm does not exist on this target)");
        }
    }
}
