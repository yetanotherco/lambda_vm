//! Host-side model of the guest `memcpy` defined in `entrypoint.rs` (the
//! `fast-memcpy` feature).
//!
//! The routine is RISC-V assembly, so it only runs inside the VM; the sweep that
//! executes the real instructions is the guest `executor/programs/rust/mem_sweep`
//! (`executor/tests/rust.rs::test_mem_sweep`). This file transliterates the
//! assembly block for block over a flat byte array, so its index arithmetic (the
//! head, the 32-byte groups, the 8-byte steps, the overlapping last doubleword and
//! the short paths) can be swept natively. It checks three things the in-VM sweep
//! can only see through guard bytes: every load stays inside the source range,
//! every store stays inside the destination range, and the stores of the 32-byte
//! loop are doubleword-aligned. A change to the assembly must be mirrored here.

use std::ops::Range;

struct Mem {
    bytes: Vec<u8>,
    src: Range<usize>,
    dst: Range<usize>,
}

impl Mem {
    fn ld(&self, addr: usize) -> u64 {
        assert!(
            addr >= self.src.start && addr + 8 <= self.src.end,
            "load of 8 at {addr} outside the source {:?}",
            self.src
        );
        u64::from_le_bytes(self.bytes[addr..addr + 8].try_into().unwrap())
    }
    fn sd(&mut self, addr: usize, value: u64) {
        assert!(
            addr >= self.dst.start && addr + 8 <= self.dst.end,
            "store of 8 at {addr} outside the destination {:?}",
            self.dst
        );
        self.bytes[addr..addr + 8].copy_from_slice(&value.to_le_bytes());
    }
    fn lbu(&self, addr: usize) -> u8 {
        assert!(
            self.src.contains(&addr),
            "byte load at {addr} outside the source {:?}",
            self.src
        );
        self.bytes[addr]
    }
    fn sb(&mut self, addr: usize, value: u8) {
        assert!(
            self.dst.contains(&addr),
            "byte store at {addr} outside the destination {:?}",
            self.dst
        );
        self.bytes[addr] = value;
    }
}

/// `memcpy(a0 = dst, a1 = src, a2 = n)`, one statement per instruction group of the
/// assembly; the numeric labels are the assembly's.
fn memcpy_model(m: &mut Mem, a0: usize, mut a1: usize, mut a2: usize) -> usize {
    let mut t0 = a0;
    if a2 >= 32 {
        // head: one doubleword wherever the destination starts, then step to the
        // next doubleword boundary of the destination (1..=8 bytes)
        let t2 = m.ld(a1);
        m.sd(t0, t2);
        let t3 = 8 - (t0 & 7);
        t0 += t3;
        a1 += t3;
        a2 -= t3;
        let t5 = a2 & !31;
        if t5 != 0 {
            let end = t0 + t5;
            // 1: four loads, then four stores
            loop {
                let w = [m.ld(a1), m.ld(a1 + 8), m.ld(a1 + 16), m.ld(a1 + 24)];
                assert_eq!(t0 % 8, 0, "the 32-byte loop stores to a misaligned address");
                for (i, w) in w.into_iter().enumerate() {
                    m.sd(t0 + 8 * i, w);
                }
                a1 += 32;
                t0 += 32;
                if t0 >= end {
                    break;
                }
            }
            a2 &= 31;
        }
        // 2 / 3: whole doublewords left
        if a2 >= 8 {
            loop {
                let t2 = m.ld(a1);
                m.sd(t0, t2);
                a1 += 8;
                t0 += 8;
                a2 -= 8;
                if a2 < 8 {
                    break;
                }
            }
        }
        // 4: the last 1..=7 bytes as one doubleword ending at the end
        if a2 != 0 {
            a1 += a2;
            t0 += a2;
            let t2 = m.ld(a1 - 8);
            m.sd(t0 - 8, t2);
        }
        return a0;
    }
    // 5: 8..=31 bytes
    if a2 >= 8 {
        let t5 = a1 + a2;
        let t6 = t0 + a2;
        // 6
        loop {
            let t2 = m.ld(a1);
            m.sd(t0, t2);
            a1 += 8;
            t0 += 8;
            a2 -= 8;
            if a2 < 8 {
                break;
            }
        }
        if a2 != 0 {
            let t2 = m.ld(t5 - 8);
            m.sd(t6 - 8, t2);
        }
        return a0;
    }
    // 7 / 8: 0..=7 bytes, one at a time
    while a2 != 0 {
        let t2 = m.lbu(a1);
        m.sb(t0, t2);
        a1 += 1;
        t0 += 1;
        a2 -= 1;
    }
    a0
}

fn pat(i: usize) -> u8 {
    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8
}

/// Copy `n` bytes from address `src` to address `dst` in a patterned memory and
/// compare the whole memory with a byte-by-byte reference copy.
fn check(size: usize, dst: usize, src: usize, n: usize) {
    let bytes: Vec<u8> = (0..size).map(pat).collect();
    let mut want = bytes.clone();
    for i in 0..n {
        want[dst + i] = bytes[src + i];
    }
    let mut m = Mem {
        bytes,
        src: src..src + n,
        dst: dst..dst + n,
    };
    let ret = memcpy_model(&mut m, dst, src, n);
    assert_eq!(ret, dst, "return value, n {n} dst {dst} src {src}");
    if let Some(at) = (0..size).find(|&i| m.bytes[i] != want[i]) {
        panic!(
            "n {n} dst%16 {} src%16 {}: byte {} (destination + {}) is wrong",
            dst % 16,
            src % 16,
            at,
            at as isize - dst as isize
        );
    }
}

#[test]
fn every_length_and_alignment_mod_8() {
    for n in 0..=1100 {
        for sa in 0..8 {
            for da in 0..8 {
                check(4096, 2048 + da, 64 + sa, n);
            }
        }
    }
}

#[test]
fn every_alignment_mod_16_around_the_thresholds() {
    for n in 0..=160 {
        for sa in 0..16 {
            for da in 0..16 {
                check(1024, 512 + da, 64 + sa, n);
            }
        }
    }
}

#[test]
fn lengths_between_the_sweep_and_a_page() {
    // Step 7 is prime to 32, so the lengths cover every remainder of the 32-byte
    // loop and of the 8-byte steps.
    for n in (1101..=4096).step_by(7) {
        for (sa, da) in [(0, 0), (1, 0), (0, 1), (7, 4)] {
            check(2 * n + 256, n + 128 + da, 64 + sa, n);
        }
    }
}

#[test]
fn large_lengths() {
    for n in [4095, 4096, 4097, 8191, 65536, 65543, 262149, 1 << 20] {
        for (sa, da) in [(0, 0), (1, 0), (0, 1), (7, 4), (3, 3)] {
            check(2 * n + 256, n + 128 + da, 64 + sa, n);
        }
    }
}

#[test]
fn same_region_and_touching_regions() {
    for n in 0..=300 {
        for a in 0..8 {
            // source == destination: the one overlap a compiler may emit
            check(1024, 64 + a, 64 + a, n);
            // destination right after the source, and right before it
            check(1024, 64 + a + n, 64 + a, n);
            check(1024, 64 + a, 64 + a + n, n);
        }
    }
}
