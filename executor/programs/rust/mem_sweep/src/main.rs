//! In-VM sweep of the guest's memory routines against values computed from the index.
//!
//! The routine under test is whatever the linker resolved `memcpy` / `memmove` /
//! `memset` / `memcmp` to in this guest. Nothing here copies bytes to build an
//! expectation: the source holds `pat(index)`, the destination is poisoned with a
//! byte `pat` never produces, and every check recomputes `pat`. So a routine cannot
//! be used to verify itself.
//!
//! Public output: `OK` + six little-endian u32 case counts (memcpy mod 8, memcpy
//! mod 16, memcpy large, memcpy same/adjacent, memmove, memset + memcmp), or `FAIL` +
//! the stage byte, length (u32), source and destination alignment, and the index
//! (u32) of the first wrong byte relative to the destination start (negative = in
//! the guard before it, encoded two's complement).

use core::hint::black_box;
use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memmove(dst: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memset(dst: *mut u8, c: i32, n: usize) -> *mut u8;
    fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32;
}

/// Bytes kept untouched on each side of a destination.
const GUARD: usize = 24;
/// A byte `pat` never returns (it only returns odd bytes).
const POISON: u8 = 0xA4;
const LARGE: usize = 1 << 20;
/// Lengths of the exhaustive sweeps.
const MAX_LEN: usize = 1100;
const MAX_LEN_MOD16: usize = 160;
const LARGE_LENS: [usize; 6] = [4095, 4096, 4097, 8191, 65536, 65543];
const LARGE_ALIGN: [usize; 4] = [0, 1, 4, 7];
/// The two largest lengths, and the lengths between `MAX_LEN` and a page, run at four
/// (source, destination) offsets only.
const HUGE_LENS: [usize; 2] = [262149, LARGE];
const HUGE_ALIGN: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (7, 4)];

/// Source pattern: an odd pseudo-random byte per index, with no short period.
#[inline(always)]
fn pat(i: usize) -> u8 {
    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8 | 1
}

struct Fail {
    stage: u8,
    len: usize,
    sa: usize,
    da: usize,
    at: isize,
}

fn fail(stage: u8, len: usize, sa: usize, da: usize, at: isize) -> Fail {
    Fail {
        stage,
        len,
        sa,
        da,
        at,
    }
}

/// Poison `[d - GUARD, d + n + GUARD)` byte by byte through volatile writes, so the
/// compiler cannot turn the loop into a call to a routine under test.
#[inline(never)]
fn poison(base: *mut u8, d: usize, n: usize) {
    for i in d - GUARD..d + n + GUARD {
        unsafe { core::ptr::write_volatile(base.add(i), POISON) };
    }
}

/// Check the guards around `[d, d + n)` and that `dst[d + i] == pat(s + i)`.
#[inline(never)]
fn check_copy(base: *const u8, d: usize, n: usize, s: usize) -> Result<(), isize> {
    for i in 0..GUARD {
        if unsafe { core::ptr::read_volatile(base.add(d - GUARD + i)) } != POISON {
            return Err(i as isize - GUARD as isize);
        }
        if unsafe { core::ptr::read_volatile(base.add(d + n + i)) } != POISON {
            return Err((n + i) as isize);
        }
    }
    for i in 0..n {
        if unsafe { core::ptr::read_volatile(base.add(d + i)) } != pat(s + i) {
            return Err(i as isize);
        }
    }
    Ok(())
}

/// Check that `src[i] == pat(i)` over the whole source (the routines must not write it).
#[inline(never)]
fn check_source(src: &[u8]) -> Result<(), isize> {
    for i in 0..src.len() {
        if unsafe { core::ptr::read_volatile(src.as_ptr().add(i)) } != pat(i) {
            return Err(i as isize);
        }
    }
    Ok(())
}

/// One `memcpy` case: poison, copy `n` bytes from `src + s` to `dst + d`, check the
/// return value, the copied bytes and the guards.
#[inline(never)]
fn copy_case(
    stage: u8,
    src: &[u8],
    dst: &mut [u8],
    s: usize,
    d: usize,
    n: usize,
) -> Result<(), Fail> {
    let base = dst.as_mut_ptr();
    poison(base, d, n);
    let to = unsafe { base.add(d) };
    let ret = unsafe { memcpy(black_box(to), black_box(src.as_ptr().add(s)), black_box(n)) };
    if ret != to {
        return Err(fail(
            stage,
            n,
            (src.as_ptr() as usize + s) % 16,
            (base as usize + d) % 16,
            isize::MIN,
        ));
    }
    check_copy(base, d, n, s).map_err(|at| {
        fail(
            stage,
            n,
            (src.as_ptr() as usize + s) % 16,
            (base as usize + d) % 16,
            at,
        )
    })
}

fn run() -> Result<[u32; 6], Fail> {
    let mut counts = [0u32; 6];
    let mut src = vec![0u8; LARGE + 64];
    for (i, b) in src.iter_mut().enumerate() {
        *b = pat(i);
    }
    let mut dst = vec![POISON; LARGE + 2 * GUARD + 64];
    // Destinations start at an 8-aligned (then 16-aligned) address plus the wanted
    // offset, whatever the allocator returned.
    let d0 = GUARD + (16 - (dst.as_ptr() as usize + GUARD) % 16) % 16;
    let s0 = (16 - src.as_ptr() as usize % 16) % 16;

    // 1. memcpy: every length up to MAX_LEN, every source and destination offset mod 8.
    for n in 0..=MAX_LEN {
        for sa in 0..8 {
            for da in 0..8 {
                copy_case(1, &src, &mut dst, s0 + sa, d0 + da, n)?;
                counts[0] += 1;
            }
        }
    }
    // 2. memcpy: every offset mod 16 for the lengths around the routine's thresholds.
    for n in 0..=MAX_LEN_MOD16 {
        for sa in 0..16 {
            for da in 0..16 {
                copy_case(2, &src, &mut dst, s0 + sa, d0 + da, n)?;
                counts[1] += 1;
            }
        }
    }
    // 3. memcpy: large lengths, and lengths up to a page.
    for &n in &LARGE_LENS {
        for &sa in &LARGE_ALIGN {
            for &da in &LARGE_ALIGN {
                copy_case(3, &src, &mut dst, s0 + sa, d0 + da, n)?;
                counts[2] += 1;
            }
        }
    }
    for &n in &HUGE_LENS {
        for &(sa, da) in &HUGE_ALIGN {
            copy_case(3, &src, &mut dst, s0 + sa, d0 + da, n)?;
            counts[2] += 1;
        }
    }
    // Lengths between the exhaustive sweep and a page, at four offset pairs. Step 7 is
    // prime to 32, so they cover every remainder of the 32-byte loop and the 8-byte steps.
    for n in (MAX_LEN + 1..=4096).step_by(7) {
        for &(sa, da) in &HUGE_ALIGN {
            copy_case(3, &src, &mut dst, s0 + sa, d0 + da, n)?;
            counts[2] += 1;
        }
    }
    // 3b. the compiler's own copies: a slice copy of run-time length and a fixed
    // 767-byte block (the size the block guest copies most).
    for da in 0..8 {
        let n = black_box(767usize);
        poison(dst.as_mut_ptr(), d0 + da, n);
        dst[d0 + da..d0 + da + n].copy_from_slice(&src[s0 + 1..s0 + 1 + n]);
        check_copy(dst.as_ptr(), d0 + da, n, s0 + 1).map_err(|at| fail(3, n, 1, da, at))?;
        poison(dst.as_mut_ptr(), d0 + da, 767);
        unsafe {
            let from = src.as_ptr().add(s0 + 3) as *const [u8; 767];
            let to = dst.as_mut_ptr().add(d0 + da) as *mut [u8; 767];
            core::ptr::copy_nonoverlapping(black_box(from), black_box(to), 1);
        }
        check_copy(dst.as_ptr(), d0 + da, 767, s0 + 3).map_err(|at| fail(3, 767, 3, da, at))?;
        counts[2] += 2;
    }
    check_source(&src).map_err(|at| fail(3, 0, 0, 0, at))?;

    // 4. memcpy inside one buffer: source == destination (the one overlap the
    // compiler may emit), and regions that touch without overlapping.
    let mut buf = vec![0u8; 4096];
    let b0 = 64 + (16 - (buf.as_ptr() as usize + 64) % 16) % 16;
    for n in 0..=300usize {
        for a in 0..8 {
            for which in 0..3 {
                for (i, b) in buf.iter_mut().enumerate().take(b0 + a + 2 * n + 64) {
                    *b = pat(i);
                }
                let p = b0 + a;
                // 0: same region; 1: destination right after the source; 2: right before.
                let (s, d) = match which {
                    0 => (p, p),
                    1 => (p, p + n),
                    _ => (p + n, p),
                };
                let base = buf.as_mut_ptr();
                let ret = unsafe {
                    memcpy(
                        black_box(base.add(d)),
                        black_box(base.add(s) as *const u8),
                        black_box(n),
                    )
                };
                if ret != unsafe { base.add(d) } {
                    return Err(fail(4, n, a, which, isize::MIN));
                }
                for j in 0..b0 + a + 2 * n + 64 {
                    let want = if j >= d && j < d + n {
                        pat(s + (j - d))
                    } else {
                        pat(j)
                    };
                    if unsafe { core::ptr::read_volatile(base.add(j)) } != want {
                        return Err(fail(4, n, a, which, j as isize - d as isize));
                    }
                }
                counts[3] += 1;
            }
        }
    }

    // 5. memmove: overlapping forward and backward, adjacent, and disjoint regions.
    const DELTAS: [isize; 17] = [
        -33, -32, -17, -16, -9, -8, -7, -1, 0, 1, 7, 8, 9, 16, 17, 32, 33,
    ];
    for n in 0..=200usize {
        for a in 0..8 {
            for k in 0..DELTAS.len() + 2 {
                let delta = match k {
                    k if k < DELTAS.len() => DELTAS[k],
                    k if k == DELTAS.len() => n as isize,
                    _ => -(n as isize),
                };
                let s = b0 + 256 + a;
                let d = (s as isize + delta) as usize;
                let lo = s.min(d) - GUARD;
                let hi = s.max(d) + n + GUARD;
                for j in lo..hi {
                    unsafe { core::ptr::write_volatile(buf.as_mut_ptr().add(j), pat(j)) };
                }
                let base = buf.as_mut_ptr();
                let ret = unsafe {
                    memmove(
                        black_box(base.add(d)),
                        black_box(base.add(s) as *const u8),
                        black_box(n),
                    )
                };
                if ret != unsafe { base.add(d) } {
                    return Err(fail(5, n, a, k, isize::MIN));
                }
                for j in lo..hi {
                    let want = if j >= d && j < d + n {
                        pat(s + (j - d))
                    } else {
                        pat(j)
                    };
                    if unsafe { core::ptr::read_volatile(base.add(j)) } != want {
                        return Err(fail(5, n, a, k, j as isize - d as isize));
                    }
                }
                counts[4] += 1;
            }
        }
    }

    // 6. memset and memcmp.
    for n in 0..=300usize {
        for a in 0..8 {
            for &c in &[0x00u8, 0x3C] {
                let d = b0 + a;
                poison(buf.as_mut_ptr(), d, n);
                let base = buf.as_mut_ptr();
                let ret =
                    unsafe { memset(black_box(base.add(d)), black_box(c as i32), black_box(n)) };
                if ret != unsafe { base.add(d) } {
                    return Err(fail(6, n, a, c as usize, isize::MIN));
                }
                for j in d - GUARD..d + n + GUARD {
                    let want = if j >= d && j < d + n { c } else { POISON };
                    if unsafe { core::ptr::read_volatile(base.add(j)) } != want {
                        return Err(fail(6, n, a, c as usize, j as isize - d as isize));
                    }
                }
                counts[5] += 1;
            }
        }
    }
    let mut other = vec![0u8; 512];
    for n in 0..=64usize {
        for a in 0..8 {
            for b in 0..8 {
                // equal contents at two alignments, then one differing byte at the
                // start, the middle and the end
                for (i, x) in other.iter_mut().enumerate().take(b + n + 8) {
                    *x = if i >= b && i < b + n {
                        pat(s0 + a + (i - b))
                    } else {
                        POISON
                    };
                }
                let pa = unsafe { src.as_ptr().add(s0 + a) };
                let pb = unsafe { other.as_ptr().add(b) };
                if unsafe { memcmp(black_box(pa), black_box(pb), black_box(n)) } != 0 {
                    return Err(fail(7, n, a, b, 0));
                }
                counts[5] += 1;
                if n == 0 {
                    continue;
                }
                for &at in &[0, n / 2, n - 1] {
                    // flip the top bit of one byte: the sign of the result is known
                    // from which side now holds the larger byte, in both argument orders
                    let keep = other[b + at];
                    other[b + at] = keep ^ 0x80;
                    let want = if keep >= 0x80 { 1i32 } else { -1i32 };
                    let ab = unsafe { memcmp(black_box(pa), black_box(pb), black_box(n)) };
                    let ba = unsafe { memcmp(black_box(pb), black_box(pa), black_box(n)) };
                    if ab.signum() != want || ba.signum() != -want {
                        return Err(fail(7, n, a, b, at as isize));
                    }
                    counts[5] += 2;
                    other[b + at] = keep;
                }
            }
        }
    }
    Ok(counts)
}

pub fn main() {
    let mut out = Vec::with_capacity(32);
    match run() {
        Ok(counts) => {
            out.extend_from_slice(b"OK");
            for c in counts {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        Err(f) => {
            out.extend_from_slice(b"FAIL");
            out.push(f.stage);
            out.extend_from_slice(&(f.len as u32).to_le_bytes());
            out.push(f.sa as u8);
            out.push(f.da as u8);
            out.extend_from_slice(&(f.at as i32).to_le_bytes());
        }
    }
    syscalls::syscalls::commit(&out);
}
