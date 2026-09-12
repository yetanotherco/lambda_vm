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
//! Two kernels. The first is one body in portable Rust, compiled twice — plain,
//! which is the correctness reference and the scalar-per-lane baseline, and
//! again inside `#[target_feature(enable = "avx512f,avx512dq")]` on `x86_64`,
//! which lets the backend vectorise the same eight lanes onto `zmm` registers.
//! The first reading of that pair, on 2026-09-12, was **0.539 ns/mul plain
//! against 1.307 with the attribute on the 9950X** and 0.756 against 1.085 on
//! the 7950X: the auto-vectoriser did not vectorise the 64×64→128 product,
//! because x86 has no such vector instruction.
//!
//! That closed auto-vectorisation, not the question, so the second kernel is
//! [`mul_reduce_x8`] — hand-written from `vpmuludq` partials with explicit
//! carries. Its arms are the `intrinsics` ones below.
//!
//! ⚠ **`#[target_feature]` deliberately, and NOT `RUSTFLAGS`.** The repo's build
//! convention forbids ad-hoc flags such as `-C target-cpu=native`: any flag
//! change forks the `sccache` key and every worktree recompiles. A function
//! attribute forks nothing, and the runtime `is_x86_feature_detected!` guard is
//! what keeps the binary portable.
//!
//! ⛔ **The reading is a ratio between arms, and two of the differences between
//! those arms are not the kernel.** A `#[target_feature]` function cannot be
//! inlined into a caller that lacks the feature, so every attributed arm pays a
//! call and a state round-trip per round that the plain arm does not; and one
//! `zmm` is ONE dependency chain where eight scalar lanes are eight, so a
//! single-register vector arm is latency-bound against a throughput-bound
//! baseline. Both push the ratio the same way. That is what the control arms in
//! [`tests::goldilocks_simd_headroom`] separate, and why the pre-registered
//! verdict is printed with them beside it rather than alone.
//!
//! | reading, intrinsics vs baseline | conclusion |
//! |---|---|
//! | **≥ 1.5×** | D′a is OPEN — the headroom is in the multiply; an 8-way structure-of-arrays permutation is worth pricing |
//! | **< 1.5×** | D′a is closed FOR THE MULTIPLY on this box. It is *not* a verdict on an 8-way permutation, whose win would come from throughput over latency — read the 4-chain control before writing that down |
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

/// The same multiply, re-spelled the way a vector unit is forced to do it: four
/// 32×32→64 partial products, an explicit carry column, then the identical
/// reduction.
///
/// ⛔ **This exists so the ALGORITHM can be proven on a machine with no
/// AVX-512.** [`mul_reduce`] gets the 128-bit product from one `mul`
/// instruction; x86 has no vector equivalent — `vpmullq` is low-half only — so
/// the vector kernel must rebuild it from four `vpmuludq` partials. That
/// reconstruction, not the folding identity, is where a vector port goes wrong,
/// and it is pure `u64` arithmetic, so it can be *checked* on this lane's
/// aarch64 laptop and merely *timed* on a box.
/// [`tests::the_lanewise_algorithm_is_the_scalar_kernel`] pins it to
/// [`mul_reduce`] over the same 71,289 pairs the field check uses. Every line
/// below is one intrinsic of [`mul_reduce_x8`], named in place.
///
/// Neither column sum can wrap: `t` is three 32-bit quantities so `t < 2^34`,
/// and `x_hi` is by construction the high word of a product of two `u64`s, so
/// it is `< 2^64` — and so is every partial sum of its non-negative terms.
fn mul_reduce_lanewise(a: u64, b: u64) -> u64 {
    const MASK32: u64 = 0xFFFF_FFFF;

    // `_mm512_mul_epu32` reads bits [31:0] of each qword and ignores the rest,
    // which is what the masks spell; the high halves are shifted down first,
    // exactly as `_mm512_srli_epi64::<32>` does.
    let a_hi = a >> 32;
    let b_hi = b >> 32;
    let ll = (a & MASK32) * (b & MASK32);
    let lh = (a & MASK32) * b_hi;
    let hl = a_hi * (b & MASK32);
    let hh = a_hi * b_hi;

    // The 32-bit carry column, then the two words of `a·b`.
    let t = (ll >> 32) + (lh & MASK32) + (hl & MASK32);
    let x_lo = (t << 32) | (ll & MASK32);
    let x_hi = hh + (t >> 32) + (lh >> 32) + (hl >> 32);

    // From here it is `mul_reduce` verbatim, with the `bool` of an
    // `overflowing_*` written as the compare-mask a vector produces instead.
    let n_hi = x_hi >> 32;
    let n_lo = x_hi & MASK32;

    let r = x_lo.wrapping_sub(n_hi);
    let borrow = u64::from(x_lo < n_hi);
    let r = r.wrapping_sub(EPSILON * borrow);

    let fold = (n_lo << 32).wrapping_sub(n_lo);
    let s = r.wrapping_add(fold);
    let carry = u64::from(s < r);
    s.wrapping_add(EPSILON * carry)
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

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_cmplt_epu64_mask, _mm512_loadu_si512,
    _mm512_mask_add_epi64, _mm512_mask_sub_epi64, _mm512_mul_epu32, _mm512_or_si512,
    _mm512_set1_epi64, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
    _mm512_xor_si512,
};

/// ★ The hand-written kernel: eight Goldilocks multiplies in one `zmm`.
///
/// The algorithm in six lines:
/// 1. split both operands at bit 32 (`vpsrlq`), because `vpmuludq` reads only
///    bits [31:0] of each qword;
/// 2. four partials `ll, lh, hl, hh` (`vpmuludq`) — the only multiply x86 has
///    that produces a full 64-bit vector result from 32-bit inputs;
/// 3. the carry column `t = (ll >> 32) + lh.lo + hl.lo`, which is `< 2^34` and
///    so cannot wrap;
/// 4. `x_lo = (t << 32) | ll.lo` and `x_hi = hh + (t >> 32) + lh.hi + hl.hi`,
///    which together are `a·b` exactly;
/// 5. fold with `2^64 ≡ 2^32 − 1`: subtract `x_hi >> 32`, then add
///    `(x_hi.lo << 32) − x_hi.lo`;
/// 6. correct each wrap with `vpcmpltuq` into a `__mmask8` and one masked
///    add/sub — the vector spelling of `mul_reduce`'s `bool as u64`.
///
/// Steps 5 and 6 ARE [`mul_reduce`]'s, lane for lane, so the output convention
/// is [`mul_reduce`]'s too: reduced modulo `p`, **not** canonicalised. That is
/// what lets the box compare the two bit for bit rather than through
/// [`GoldilocksField::canonical`], and [`mul_reduce_lanewise`] is this same
/// function written on `u64`s, checked against [`mul_reduce`] everywhere —
/// including on machines with no AVX-512 at all.
///
/// Consulted: the `2^64 ≡ 2^32 − 1` folding identity and the mask-for-branch
/// correction are the standard Goldilocks idiom, the one Plonky2's and
/// Plonky3's AVX-512 backends also use; the four-partial reconstruction is
/// plain schoolbook. This is written from the identity, not transcribed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
fn mul_reduce_x8(a: __m512i, b: __m512i) -> __m512i {
    let mask32 = _mm512_set1_epi64(0xFFFF_FFFFu64 as i64);
    let eps = _mm512_set1_epi64(EPSILON as i64);

    // 1–2: the four 32×32→64 partial products.
    let a_hi = _mm512_srli_epi64::<32>(a);
    let b_hi = _mm512_srli_epi64::<32>(b);
    let ll = _mm512_mul_epu32(a, b);
    let lh = _mm512_mul_epu32(a, b_hi);
    let hl = _mm512_mul_epu32(a_hi, b);
    let hh = _mm512_mul_epu32(a_hi, b_hi);

    // 3–4: the carry column, then the two words of the 128-bit product.
    let t = _mm512_add_epi64(
        _mm512_srli_epi64::<32>(ll),
        _mm512_add_epi64(_mm512_and_si512(lh, mask32), _mm512_and_si512(hl, mask32)),
    );
    let x_lo = _mm512_or_si512(_mm512_slli_epi64::<32>(t), _mm512_and_si512(ll, mask32));
    let x_hi = _mm512_add_epi64(
        _mm512_add_epi64(hh, _mm512_srli_epi64::<32>(t)),
        _mm512_add_epi64(_mm512_srli_epi64::<32>(lh), _mm512_srli_epi64::<32>(hl)),
    );

    // 5–6: the fold, with each wrap corrected under a compare-mask.
    let n_hi = _mm512_srli_epi64::<32>(x_hi);
    let n_lo = _mm512_and_si512(x_hi, mask32);

    let r = _mm512_sub_epi64(x_lo, n_hi);
    let borrow = _mm512_cmplt_epu64_mask(x_lo, n_hi);
    let r = _mm512_mask_sub_epi64(r, borrow, r, eps);

    let fold = _mm512_sub_epi64(_mm512_slli_epi64::<32>(n_lo), n_lo);
    let s = _mm512_add_epi64(r, fold);
    let carry = _mm512_cmplt_epu64_mask(s, r);
    _mm512_mask_add_epi64(s, carry, s, eps)
}

/// [`mul_reduce_x8`] behind the same signature and the same call shape as
/// [`mul_batch_avx512`], so the third timed arm differs from the second in the
/// KERNEL and in nothing else.
///
/// # Safety
///
/// The caller must have checked `is_x86_feature_detected!("avx512f")` and
/// `("avx512dq")`. The loads are unaligned forms over a `[u64; 8]`, which is
/// exactly one 512-bit register, so they impose no alignment requirement.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn mul_batch_avx512_intrinsics(state: &mut [u64; LANES], k: &[u64; LANES]) {
    // SAFETY: as documented above — features checked by the caller, and
    // `[u64; LANES]` is 64 bytes, read and written by unaligned forms.
    unsafe {
        let a = _mm512_loadu_si512(state.as_ptr().cast());
        let b = _mm512_loadu_si512(k.as_ptr().cast());
        _mm512_storeu_si512(state.as_mut_ptr().cast(), mul_reduce_x8(a, b));
    }
}

/// ⚠ The control the first three arms need: the SAME kernels with the round
/// loop moved INSIDE the feature boundary, and with `CHAINS` independent
/// accumulators.
///
/// Two biases make an arm-to-arm ratio uninterpretable without this, and both
/// push the same way — against the vector:
///
/// * **The call boundary.** A `#[target_feature]` function cannot be inlined
///   into a caller that lacks the feature, so `mul_batch_avx512{,_intrinsics}`
///   pay a call plus a store/load round-trip of the state on EVERY round, while
///   the plain arm is `#[inline(always)]` into the closure and keeps the state
///   in registers. That is a per-round cost the plain arm never pays, on a
///   kernel whose whole budget is a couple of nanoseconds.
/// * **The chain multiplicity.** `mul_batch` is eight INDEPENDENT scalar chains
///   — one per lane — so the plain arm is throughput-bound and fills the ports.
///   The vector arm is one `zmm`, so at `CHAINS = 1` it is a single dependency
///   chain and is LATENCY-bound. A real permutation has twelve independent
///   state lanes per round, so `CHAINS > 1` is the representative shape, not
///   the generous one.
///
/// `rounds · CHAINS` multiplies of work are done per lane, so every arm is
/// called with `ROUNDS / CHAINS` and reports over the same total.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn mul_chain_avx512_intrinsics<const CHAINS: usize>(
    state: &mut [u64; LANES],
    k: &[u64; LANES],
    rounds: usize,
) {
    // SAFETY: features checked by the caller; unaligned loads over `[u64; 8]`.
    unsafe {
        let kv = _mm512_loadu_si512(k.as_ptr().cast());
        let base = _mm512_loadu_si512(state.as_ptr().cast());
        // De-correlated per chain so the optimiser cannot collapse them into
        // one, and folded back below so none of them can be dropped.
        let mut s = [base; CHAINS];
        for (j, sj) in s.iter_mut().enumerate() {
            *sj = _mm512_add_epi64(*sj, _mm512_set1_epi64(j as i64));
        }
        for _ in 0..rounds {
            for sj in s.iter_mut() {
                *sj = mul_reduce_x8(*sj, kv);
            }
        }
        let acc = s.iter().skip(1).fold(s[0], |a, b| _mm512_xor_si512(a, *b));
        _mm512_storeu_si512(state.as_mut_ptr().cast(), acc);
    }
}

/// [`mul_chain_avx512_intrinsics`]'s scalar counterpart: the plain kernel, the
/// same round loop, the same `CHAINS` accumulators and the same fold, so the
/// two differ only in which kernel runs.
fn mul_chain_plain<const CHAINS: usize>(state: &mut [u64; LANES], k: &[u64; LANES], rounds: usize) {
    let mut s = [*state; CHAINS];
    for (j, sj) in s.iter_mut().enumerate() {
        for x in sj.iter_mut() {
            *x = x.wrapping_add(j as u64);
        }
    }
    for _ in 0..rounds {
        for sj in s.iter_mut() {
            mul_batch(sj, k);
        }
    }
    let mut acc = s[0];
    for sj in s.iter().skip(1) {
        for (a, b) in acc.iter_mut().zip(sj.iter()) {
            *a ^= *b;
        }
    }
    *state = acc;
}

/// [`mul_chain_plain`] with AVX-512 enabled for it, which is what isolates the
/// call boundary: this and [`mul_batch_avx512`] compile the same kernel with
/// the same features, and differ only in where the round loop sits.
///
/// # Safety
///
/// The caller must have checked `is_x86_feature_detected!("avx512f")` and
/// `("avx512dq")`. The body is ordinary safe Rust.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn mul_chain_plain_avx512<const CHAINS: usize>(
    state: &mut [u64; LANES],
    k: &[u64; LANES],
    rounds: usize,
) {
    mul_chain_plain::<CHAINS>(state, k, rounds);
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

    /// The 267 values whose 71,289 ordered pairs EVERY arm is checked over —
    /// one list, so "checked over the same pairs" is a fact about the code and
    /// not a claim in a report.
    ///
    /// The corners are deliberate: `p − 1` squared, `2^32` (where the folding
    /// identity switches limbs), and values whose product's high word forces
    /// both the borrow and the carry path. The rest is a deterministic spread,
    /// so a regression is reproducible.
    fn corner_values() -> Vec<u64> {
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
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..256 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            corners.push(seed);
        }
        corners
    }

    /// ★★ The gate on the KERNEL, not on the timing: `mul_reduce` must be the
    /// field's multiplication.
    ///
    /// Runs everywhere, including this lane's aarch64 laptop, because the VALUES
    /// are architecture-independent even though the speed is not. It is what
    /// makes the benchmark below unable to report a number from a wrong kernel —
    /// the failure mode that would otherwise open or close D′a on a lie.
    ///
    /// The corner list is [`corner_values`]; it is shared with every other
    /// check in this file.
    #[test]
    fn the_kernel_is_the_field_multiplication() {
        let corners = corner_values();
        let mut pairs = 0usize;
        for &a in &corners {
            for &b in &corners {
                let got = canon(mul_reduce(a, b));
                let want = (FE::from(a) * FE::from(b)).canonical_u64();
                assert_eq!(
                    got, want,
                    "mul_reduce({a:#x}, {b:#x}) = {got:#x}, field says {want:#x}"
                );
                pairs += 1;
            }
        }
        println!("mul_reduce vs the field:          {pairs}/{pairs} pairs agree");
    }

    /// ★★ The gate on the VECTOR ALGORITHM, run on a machine that cannot
    /// execute it.
    ///
    /// The box will check [`mul_reduce_x8`] itself before it times anything.
    /// This check is what makes that one a formality rather than the first time
    /// anyone has looked: [`mul_reduce_lanewise`] performs the same four
    /// partial products, the same carry column and the same two masked
    /// corrections on `u64`s, so a mistake in the RECONSTRUCTION of the
    /// 128-bit product — the part `mul_reduce` gets free from one `mul`
    /// instruction and a vector unit does not — fails here, on aarch64, before
    /// a box is ever booked.
    ///
    /// Compared bit for bit, not through [`canon`]: both sides share
    /// `mul_reduce`'s un-canonicalised convention, and comparing canonical
    /// forms would hide a divergence in the `[p, 2^64)` representatives that a
    /// later consumer could still see.
    #[test]
    fn the_lanewise_algorithm_is_the_scalar_kernel() {
        let corners = corner_values();
        let mut pairs = 0usize;
        for &a in &corners {
            for &b in &corners {
                let want = mul_reduce(a, b);
                let got = mul_reduce_lanewise(a, b);
                assert_eq!(
                    got, want,
                    "mul_reduce_lanewise({a:#x}, {b:#x}) = {got:#x}, mul_reduce says {want:#x}"
                );
                pairs += 1;
            }
        }
        println!("corner values:                    {}", corners.len());
        println!("lane-wise vs mul_reduce:          {pairs}/{pairs} pairs agree, bit for bit");
    }

    /// The chain helpers are bookkeeping around [`mul_batch`], so at one chain
    /// they must be exactly the round loop they replace — otherwise the timed
    /// controls would be measuring a different amount of work.
    #[test]
    fn the_chain_is_the_batch_repeated() {
        const ROUNDS: usize = 37;
        let k: [u64; LANES] = core::array::from_fn(|i| 0xDEAD_BEEF - i as u64);
        let start: [u64; LANES] = core::array::from_fn(|i| 0x1234_5678 + i as u64);

        let mut want = start;
        for _ in 0..ROUNDS {
            mul_batch(&mut want, &k);
        }

        let mut got = start;
        mul_chain_plain::<1>(&mut got, &k, ROUNDS);
        assert_eq!(got, want);
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

    /// ⛔ [`mul_reduce_x8`] against [`mul_reduce`] over [`corner_values`]'s
    /// 71,289 ordered pairs, eight at a time, through the very entry point the
    /// timing arm calls. Returns the pair count; panics on the first
    /// disagreement.
    ///
    /// This is the half of the correctness that only a box can run. Its scalar
    /// twin, [`the_lanewise_algorithm_is_the_scalar_kernel`], has already
    /// pinned the ALGORITHM on every machine including aarch64; this pins the
    /// INTRINSICS to the same values.
    ///
    /// The tail is padded to a whole register with `(0, 0)` — checked like any
    /// other pair, and not counted.
    #[cfg(target_arch = "x86_64")]
    fn intrinsics_agree_with_mul_reduce() -> usize {
        let corners = corner_values();
        let mut lhs: Vec<u64> = Vec::new();
        let mut rhs: Vec<u64> = Vec::new();
        for &a in &corners {
            for &b in &corners {
                lhs.push(a);
                rhs.push(b);
            }
        }
        let pairs = lhs.len();
        while lhs.len() % LANES != 0 {
            lhs.push(0);
            rhs.push(0);
        }

        for (ac, bc) in lhs.chunks_exact(LANES).zip(rhs.chunks_exact(LANES)) {
            let a: [u64; LANES] = ac.try_into().expect("chunks_exact yields LANES");
            let k: [u64; LANES] = bc.try_into().expect("chunks_exact yields LANES");
            let want: [u64; LANES] = core::array::from_fn(|i| mul_reduce(a[i], k[i]));
            let mut got = a;
            // SAFETY: the caller only reaches this after detecting avx512f and
            // avx512dq.
            unsafe { mul_batch_avx512_intrinsics(&mut got, &k) };
            assert_eq!(
                got, want,
                "mul_reduce_x8 disagrees with mul_reduce on a={a:#x?} b={k:#x?}"
            );
        }
        pairs
    }

    /// ★ The D′a reading. `#[ignore]`d because it is a timing measurement, not
    /// a property.
    ///
    /// Every arm runs the same chained multiply over the same lane count and
    /// reports over the same total number of multiplies. The ratio is the
    /// load-bearing output and the absolutes are not, exactly as `rpx::ladder`
    /// records for the hash candidates.
    ///
    /// Three arms answer the pre-registered question and four are controls:
    ///
    /// | arm | kernel | called per round | chains |
    /// |---|---|---|---|
    /// | baseline | scalar | no, inlined | 8 scalar |
    /// | avx512f,avx512dq | scalar, auto-vectorised | yes | 8 scalar |
    /// | intrinsics | [`mul_reduce_x8`] | yes | 1 vector |
    /// | plain, loop in the feature | scalar | no | 8 scalar |
    /// | intrinsics, loop in the feature | [`mul_reduce_x8`] | no | 1 vector |
    /// | plain ×4 | scalar | no | 32 scalar |
    /// | intrinsics ×4 | [`mul_reduce_x8`] | no | 4 vector |
    ///
    /// ⚠ The pre-registered rule reads the THIRD arm against the first, and is
    /// printed as such. The controls are printed beside it because those two
    /// arms differ in two things besides the kernel — a per-round call the
    /// baseline does not pay, and a chain multiplicity that leaves the vector
    /// latency-bound while the scalar arm is throughput-bound — and both push
    /// the ratio the same way. See [`mul_chain_avx512_intrinsics`].
    #[test]
    #[ignore]
    fn goldilocks_simd_headroom() {
        const ROUNDS: usize = 4_000_000;
        const REPEATS: usize = 5;
        const CHAINS: usize = 4;

        // ⛔ The kernel check runs BEFORE a single stopwatch starts, so a timing
        // line from a wrong kernel is not a reachable state.
        #[cfg(target_arch = "x86_64")]
        let avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq");
        #[cfg(target_arch = "x86_64")]
        let checked = if avx512 {
            Some(intrinsics_agree_with_mul_reduce())
        } else {
            None
        };

        println!("Goldilocks multiply, 8 lanes, chained — this machine:");
        let k: [u64; LANES] = core::array::from_fn(|i| 0xC0FF_EE00_0000_0001 + i as u64);

        let base = time("baseline (no target_feature)", ROUNDS, REPEATS, |s| {
            for _ in 0..ROUNDS {
                mul_batch(s, &k);
            }
        });

        #[cfg(target_arch = "x86_64")]
        {
            if avx512 {
                let simd = time("avx512f,avx512dq", ROUNDS, REPEATS, |s| {
                    for _ in 0..ROUNDS {
                        // SAFETY: both features were detected above.
                        unsafe { mul_batch_avx512(s, &k) };
                    }
                });
                let intr = time("intrinsics avx512f,avx512dq", ROUNDS, REPEATS, |s| {
                    for _ in 0..ROUNDS {
                        // SAFETY: both features were detected above.
                        unsafe { mul_batch_avx512_intrinsics(s, &k) };
                    }
                });

                println!("  -- controls: round loop inside the feature, no per-round call --");
                let plain_in = time("plain, loop in the feature", ROUNDS, REPEATS, |s| {
                    // SAFETY: both features were detected above.
                    unsafe { mul_chain_plain_avx512::<1>(s, &k, ROUNDS) };
                });
                let intr_in = time("intrinsics, loop in the feature", ROUNDS, REPEATS, |s| {
                    // SAFETY: both features were detected above.
                    unsafe { mul_chain_avx512_intrinsics::<1>(s, &k, ROUNDS) };
                });
                let plain_c = time("plain, 4 chains", ROUNDS, REPEATS, |s| {
                    // SAFETY: both features were detected above.
                    unsafe { mul_chain_plain_avx512::<CHAINS>(s, &k, ROUNDS / CHAINS) };
                });
                let intr_c = time("intrinsics, 4 chains", ROUNDS, REPEATS, |s| {
                    // SAFETY: both features were detected above.
                    unsafe { mul_chain_avx512_intrinsics::<CHAINS>(s, &k, ROUNDS / CHAINS) };
                });

                println!();
                let pairs = checked.expect("the kernel check runs whenever the arms do");
                println!(
                    "  kernel check: {pairs}/{pairs} pairs agree with mul_reduce, bit for bit"
                );
                println!();

                // ── the pre-registered reading, on the pre-registered arms.
                let ratio = base / intr;
                println!("  ⇒ PRE-REGISTERED  intrinsics vs baseline: {ratio:.2}×  (rule: ≥ 1.5×)");
                println!(
                    "  ⇒ D′a is {}",
                    if ratio >= 1.5 {
                        "OPEN — the headroom is in the multiply; stage 2 (an 8-way \
                         structure-of-arrays permutation) is worth pricing"
                    } else {
                        "CLOSED FOR THE MULTIPLY on this box — which is NOT a verdict on an \
                         8-way permutation, whose win would come from throughput over latency"
                    }
                );

                // ── the controls, which the rule above does not read.
                println!();
                println!("  controls (NOT the pre-registered rule):");
                println!(
                    "  ⇒ the per-round call boundary costs the vector arm {:.3} ns/mul, \
                     and the scalar arm {:.3}",
                    intr - intr_in,
                    simd - plain_in,
                );
                println!(
                    "  ⇒ auto-vectorisation alone, call-free:    {:.2}× the baseline",
                    base / plain_in
                );
                println!(
                    "  ⇒ call-free, 1 chain   intrinsics vs plain: {:.2}×",
                    plain_in / intr_in
                );
                println!(
                    "  ⇒ call-free, 4 chains  intrinsics vs plain: {:.2}×  ← the shape a \
                     permutation has",
                    plain_c / intr_c
                );

                // What the prize would be, at lane E's measured shares: the
                // permutation is ~83% of `execute`, and `execute` is 1.82 s of a
                // serial wrap. Taken at the best CALL-FREE ratio, because a
                // rewritten permutation would not re-enter the feature per
                // multiply.
                let best = (plain_c / intr_c).max(plain_in / intr_in).max(1.0);
                println!(
                    "  ⇒ at 83% of a 1.82 s wrap executor and {best:.2}×: {:.2} s → {:.2} s",
                    1.82,
                    1.82 * (0.17 + 0.83 / best),
                );
            } else {
                println!("  (no AVX-512 on this machine — the SIMD arms did not run)");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = base;
            let _ = CHAINS;
            println!("  (not x86_64 — the SIMD arms do not exist on this target)");
        }
    }
}
