//! Exhaustive MDS check of a circulant matrix over Goldilocks: every square
//! submatrix must be nonsingular mod p.
//!
//! A circulant `M[i][j] = row[(j - i) mod t]` is invariant under shifting the
//! row and column index sets together, so every submatrix equals one whose row
//! set contains 0. We enumerate those row sets (sum_k C(t-1,k-1)·C(t,k)
//! submatrices; C(31,15) ≈ 3.0e8 at t = 16) and take each determinant by
//! Gaussian elimination mod p.
//!
//! Build and run (no cargo; std only):
//!   rustc -O -C opt-level=3 scripts/poseidon1/mds_check.rs -o /tmp/mds_check
//!   /tmp/mds_check 16 1,1,51,1,11,17,2,1,101,63,15,2,67,22,13,3
//! Prints one line per size k and a final `MDS: yes|no`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Goldilocks reduction of a 128-bit product (2^64 = 2^32 - 1, 2^96 = -1 mod p).
fn mul(a: u64, b: u64) -> u64 {
    let x = a as u128 * b as u128;
    let (lo, hi) = (x as u64, (x >> 64) as u64);
    let (hi_hi, hi_lo) = (hi >> 32, hi & 0xFFFF_FFFF);
    let (mut t0, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        t0 = t0.wrapping_sub(0xFFFF_FFFF);
    }
    let t1 = hi_lo * 0xFFFF_FFFF;
    let (mut r, carry) = t0.overflowing_add(t1);
    if carry {
        r = r.wrapping_add(0xFFFF_FFFF);
    }
    if r >= P { r - P } else { r }
}

fn sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + (P - b) }
}

/// True iff the k×k matrix `m` (row-major, stride 16) is nonsingular mod p.
/// Division-free elimination: row_r <- piv·row_r - m[r][c]·row_c keeps the
/// rank, so the matrix is nonsingular iff every column finds a nonzero pivot.
fn nonsingular(m: &mut [u64; 256], k: usize) -> bool {
    for c in 0..k {
        let Some(piv) = (c..k).find(|&r| m[r * 16 + c] != 0) else {
            return false;
        };
        if piv != c {
            for j in 0..k {
                m.swap(piv * 16 + j, c * 16 + j);
            }
        }
        let pv = m[c * 16 + c];
        for r in c + 1..k {
            let f = m[r * 16 + c];
            if f == 0 {
                continue;
            }
            for j in c..k {
                m[r * 16 + j] = sub(mul(pv, m[r * 16 + j]), mul(f, m[c * 16 + j]));
            }
        }
    }
    true
}

/// Subsets of {0..t} of size k, as index vectors.
fn subsets(t: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for mask in 0u32..(1 << t) {
        if mask.count_ones() as usize == k {
            out.push((0..t).filter(|&i| mask >> i & 1 == 1).collect());
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args[1].parse().unwrap();
    let row: Vec<u64> = args[2].split(',').map(|s| s.parse().unwrap()).collect();
    assert_eq!(row.len(), t);
    assert!(t <= 16);
    let m = |i: usize, j: usize| row[(j + t - i) % t] % P;
    let threads = std::env::var("THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    let mut all_ok = true;
    let mut total = 0u64;
    for k in 1..=t {
        let rows: Vec<Vec<usize>> = subsets(t, k).into_iter().filter(|r| r[0] == 0).collect();
        let cols = subsets(t, k);
        let bad = AtomicU64::new(0);
        let first_bad = std::sync::Mutex::new(None::<(Vec<usize>, Vec<usize>)>);
        let next = AtomicU64::new(0);
        thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut buf = [0u64; 256];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed) as usize;
                        if i >= rows.len() {
                            break;
                        }
                        let r = &rows[i];
                        for c in &cols {
                            for (a, &ri) in r.iter().enumerate() {
                                for (b, &cj) in c.iter().enumerate() {
                                    buf[a * 16 + b] = m(ri, cj);
                                }
                            }
                            if !nonsingular(&mut buf, k) {
                                bad.fetch_add(1, Ordering::Relaxed);
                                let mut fb = first_bad.lock().unwrap();
                                if fb.is_none() {
                                    *fb = Some((r.clone(), c.clone()));
                                }
                            }
                        }
                    }
                });
            }
        });
        let n = rows.len() as u64 * cols.len() as u64;
        total += n;
        let b = bad.load(Ordering::Relaxed);
        all_ok &= b == 0;
        println!(
            "k={k:2}: {n:>11} submatrices (rows containing 0), singular {b}{}",
            first_bad
                .into_inner()
                .unwrap()
                .map(|(r, c)| format!("; first rows {r:?} cols {c:?}"))
                .unwrap_or_default()
        );
    }
    println!("t={t} total {total} submatrices; MDS: {}", if all_ok { "yes" } else { "no" });
}
