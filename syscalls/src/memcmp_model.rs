//! Host-side model of the guest `memcmp` defined in `entrypoint.rs`.
//!
//! The routine is RISC-V assembly, so it only runs inside the VM; the sweep that
//! executes the real instructions is the guest `executor/programs/rust/mem_sweep`
//! (`executor/tests/rust.rs::test_mem_sweep`). This file transliterates the
//! assembly block for block over a flat byte array, so its index arithmetic (the
//! first byte, the doubleword loop, the overlapping last doubleword, the search
//! inside a differing doubleword and the short path) can be swept natively against
//! a byte-by-byte reference. Every load is checked against its operand's range,
//! which the in-VM sweep can only see through guard bytes. A change to the assembly
//! must be mirrored here.

use std::ops::Range;

struct Mem {
    bytes: Vec<u8>,
    a: Range<usize>,
    b: Range<usize>,
}

impl Mem {
    fn in_range(&self, addr: usize, len: usize) {
        let inside = |r: &Range<usize>| addr >= r.start && addr + len <= r.end;
        assert!(
            inside(&self.a) || inside(&self.b),
            "load of {len} at {addr} outside both operands {:?} {:?}",
            self.a,
            self.b
        );
    }
    fn ld(&self, addr: usize) -> u64 {
        self.in_range(addr, 8);
        u64::from_le_bytes(self.bytes[addr..addr + 8].try_into().unwrap())
    }
    fn lbu(&self, addr: usize) -> u64 {
        self.in_range(addr, 1);
        u64::from(self.bytes[addr])
    }
}

/// `memcmp(a0 = a, a1 = b, a2 = n)`, one statement per instruction group of the
/// assembly; the comments name the assembly's labels. Returns `a0`.
fn memcmp_model(m: &Mem, mut a0: usize, mut a1: usize, mut a2: usize) -> i64 {
    if a2 == 0 {
        return 0; // .Lmemcmp_equal
    }
    let (mut a3, mut a4) = (m.lbu(a0), m.lbu(a1));
    if a3 != a4 {
        return a3 as i64 - a4 as i64; // .Lmemcmp_byte_diff
    }
    let (mut t1, mut t2);
    if a2 < 8 {
        // .Lmemcmp_bytes
        loop {
            a0 += 1;
            a1 += 1;
            a2 -= 1;
            if a2 == 0 {
                return 0;
            }
            a3 = m.lbu(a0);
            a4 = m.lbu(a1);
            if a3 != a4 {
                return a3 as i64 - a4 as i64;
            }
        }
    }
    let a5 = a0 + a2 - 8;
    let a6 = a1 + a2 - 8;
    'diff: {
        // .Lmemcmp_words
        loop {
            t1 = m.ld(a0);
            t2 = m.ld(a1);
            if t1 != t2 {
                break 'diff;
            }
            a0 += 8;
            a1 += 8;
            if a0 >= a5 {
                break;
            }
        }
        t1 = m.ld(a5);
        t2 = m.ld(a6);
        if t1 != t2 {
            break 'diff;
        }
        return 0; // .Lmemcmp_equal
    }
    // .Lmemcmp_word_diff
    loop {
        a3 = t1 & 0xff;
        a4 = t2 & 0xff;
        if a3 != a4 {
            return a3 as i64 - a4 as i64;
        }
        t1 >>= 8;
        t2 >>= 8;
    }
}

fn reference(x: &[u8], y: &[u8]) -> i64 {
    x.iter()
        .zip(y)
        .find(|(p, q)| p != q)
        .map_or(0, |(&p, &q)| i64::from(p) - i64::from(q))
}

fn pat(i: usize) -> u8 {
    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8
}

/// Operands of `n` bytes at `64 + ao` (`a`) and `512 + bo` (`b`), holding the same
/// pattern.
fn operands(n: usize, ao: usize, bo: usize) -> Mem {
    let (a, b) = (64 + ao, 512 + bo);
    let mut bytes: Vec<u8> = (0..1024).map(pat).collect();
    for i in 0..n {
        bytes[b + i] = bytes[a + i];
    }
    Mem {
        bytes,
        a: a..a + n,
        b: b..b + n,
    }
}

/// Apply each (position, xor) to `b`, check the model against the reference in both
/// argument orders, and undo the flips.
fn check(m: &mut Mem, flips: &[(usize, u8)]) {
    let (a, b, n) = (m.a.start, m.b.start, m.a.len());
    for &(at, x) in flips {
        m.bytes[b + at] ^= x;
    }
    let want = reference(&m.bytes[a..a + n], &m.bytes[b..b + n]);
    let what = || format!("n {n} a%8 {} b%8 {} flips {flips:?}", a % 8, b % 8);
    assert_eq!(memcmp_model(m, a, b, n), want, "{}", what());
    assert_eq!(memcmp_model(m, b, a, n), -want, "{}, swapped", what());
    for &(at, x) in flips {
        m.bytes[b + at] ^= x;
    }
}

/// Every differing position up to 80 bytes (every remainder mod 8 of the length and of
/// the position, the overlapping last doubleword included); above that the positions
/// near the start, the middle and the end.
fn positions(n: usize) -> Vec<usize> {
    if n <= 80 {
        return (0..n).collect();
    }
    let mut at: Vec<usize> = (0..10)
        .chain(n / 2 - 8..n / 2 + 8)
        .chain(n - 17..n)
        .collect();
    at.dedup();
    at
}

#[test]
fn every_length_alignment_and_differing_position() {
    for n in 0..=300 {
        for ao in 0..8 {
            for bo in 0..8 {
                let mut m = operands(n, ao, bo);
                check(&mut m, &[]);
                for at in positions(n) {
                    for x in [0x80, 0x01, 0xff] {
                        check(&mut m, &[(at, x)]);
                    }
                    // a later difference of the opposite sense must not decide
                    if at + 1 < n {
                        check(&mut m, &[(at, 0x01), (n - 1, 0x80)]);
                    }
                }
            }
        }
    }
}

#[test]
fn same_operand() {
    for n in 0..=300 {
        for ao in 0..8 {
            let mut m = operands(n, ao, ao);
            m.b = m.a.clone();
            let a = m.a.start;
            assert_eq!(memcmp_model(&m, a, a, n), 0, "n {n} a%8 {ao}");
        }
    }
}
