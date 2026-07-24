//! Software baseline for the FEXT accelerator benchmark: computes N iterations
//! of `a = a*b + c` over the degree-3 Goldilocks extension `Fp[x]/(x^3 - 2)` in
//! plain RISC-V (no accelerator). Compared against `fext_bench.s` (same N via
//! FEXT_FMA) to measure the accelerator's proving-cost benefit.
use core::hint::black_box;
use lambda_vm_syscalls as syscalls;

const P: u64 = 0xFFFF_FFFF_0000_0001; // Goldilocks prime, 2^64 - 2^32 + 1.

/// Reduce a 128-bit product to a Goldilocks field element using
/// `2^64 ≡ 2^32 - 1` and `2^96 ≡ -1 (mod p)`. Representative of the real
/// reduction's instruction cost (overflow-safe).
#[inline(always)]
fn gold_reduce(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hi_hi = hi >> 32;
    let hi_lo = hi & 0xFFFF_FFFF;

    let (a, borrow) = lo.overflowing_sub(hi_hi);
    let a = if borrow { a.wrapping_add(P) } else { a };
    let t = hi_lo.wrapping_mul(0xFFFF_FFFF);
    let (r, carry) = a.overflowing_add(t);
    if carry { r.wrapping_sub(P) } else { r }
}

#[inline(always)]
fn gmul(a: u64, b: u64) -> u64 {
    gold_reduce((a as u128) * (b as u128))
}

#[inline(always)]
fn gadd(a: u64, b: u64) -> u64 {
    let s = a as u128 + b as u128;
    if s >= P as u128 { (s - P as u128) as u64 } else { s as u64 }
}

/// `a*b + c` over Fp3 with `w^3 = 2` (same formula as the FEXT_FMA chip).
#[inline(always)]
fn fp3_fma(a: [u64; 3], b: [u64; 3], c: [u64; 3]) -> [u64; 3] {
    let dbl = |x| gadd(x, x);
    let o0 = gadd(
        gadd(gmul(a[0], b[0]), dbl(gadd(gmul(a[1], b[2]), gmul(a[2], b[1])))),
        c[0],
    );
    let o1 = gadd(
        gadd(gadd(gmul(a[0], b[1]), gmul(a[1], b[0])), dbl(gmul(a[2], b[2]))),
        c[1],
    );
    let o2 = gadd(
        gadd(gadd(gmul(a[0], b[2]), gmul(a[1], b[1])), gmul(a[2], b[0])),
        c[2],
    );
    [o0, o1, o2]
}

pub fn main() {
    // black_box prevents constant-folding; the runtime loop bound prevents
    // unrolling; feeding `a` back makes each iteration depend on the previous.
    let mut a = [black_box(1u64), black_box(2), black_box(3)];
    let b = [black_box(4u64), black_box(5), black_box(6)];
    let c = [black_box(7u64), black_box(8), black_box(9)];

    let n = black_box(4096u32);
    let mut i = 0u32;
    while i < n {
        a = fp3_fma(black_box(a), b, c);
        i += 1;
    }

    // Commit the result so the loop is observable (not dead-code eliminated).
    let mut out = [0u8; 24];
    out[0..8].copy_from_slice(&a[0].to_le_bytes());
    out[8..16].copy_from_slice(&a[1].to_le_bytes());
    out[16..24].copy_from_slice(&a[2].to_le_bytes());
    syscalls::syscalls::commit(&out);
}
