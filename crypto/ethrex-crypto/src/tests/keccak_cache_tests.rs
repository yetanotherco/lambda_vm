//! Host-side tests for the guest's Keccak-256 cache (`keccak_cache`), with ethrex's
//! reference `keccak_hash` as the hash behind it.

use crate::keccak_cache::{KeccakCache, EMPTY_DIGEST};
use ethrex_crypto::keccak::keccak_hash;

/// Look `input` up and check the digest against the reference; returns whether the
/// cache had to hash (a miss).
fn lookup(cache: &mut KeccakCache, input: &[u8]) -> bool {
    let mut hashed = false;
    let got = cache.keccak256(input, |x| {
        hashed = true;
        keccak_hash(x)
    });
    assert_eq!(
        got,
        keccak_hash(input),
        "{}-byte input {input:02x?}",
        input.len()
    );
    hashed
}

fn pattern(len: usize, seed: u64) -> Vec<u8> {
    (0..len as u64)
        .map(|i| {
            ((i + 1)
                .wrapping_mul(seed | 1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                >> 56) as u8
        })
        .collect()
}

#[test]
fn empty_digest_is_keccak_of_nothing() {
    assert_eq!(EMPTY_DIGEST, keccak_hash([]));
}

#[test]
fn every_length_to_three_blocks_twice() {
    let mut cache = KeccakCache::new();
    for round in 0..2 {
        for len in 0..=410 {
            let hashed = lookup(&mut cache, &pattern(len, len as u64));
            let cached = matches!(len, 0 | 20 | 32 | 64);
            // The empty input never hashes; the cached lengths hash only the first
            // time; every other length hashes every time.
            let want = len != 0 && !(cached && round == 1);
            assert_eq!(hashed, want, "length {len} round {round}");
        }
    }
}

/// Two inputs that differ in doublewords `i` and `j` by the same value fold to the
/// same slot. Alternating them must replace the entry each time and never return
/// the other one's digest, whichever pair of words differs; for 20 bytes the delta
/// stays within the last word's four bytes.
#[test]
fn same_slot_inputs_never_share_a_digest() {
    for (len, words) in [(20usize, 3usize), (32, 4), (64, 8)] {
        for i in 0..words {
            for j in i + 1..words {
                let mut cache = KeccakCache::new();
                let a = pattern(len, (len * 100 + i * 10 + j) as u64);
                let mut b = a.clone();
                // only the bytes both words have, so that the two folds stay equal
                let bytes = (len - 8 * j).min(8);
                let delta = 0x0102_0304_a5c3_5a3cu64 & (u64::MAX >> (64 - 8 * bytes));
                for w in [i, j] {
                    for k in 0..bytes {
                        b[8 * w + k] ^= (delta >> (8 * k)) as u8;
                    }
                }
                for _ in 0..3 {
                    assert!(
                        lookup(&mut cache, &a),
                        "len {len} words {i} {j}: a must miss"
                    );
                    assert!(
                        lookup(&mut cache, &b),
                        "len {len} words {i} {j}: b must miss"
                    );
                }
            }
        }
    }
}

/// A one-byte change anywhere in a cached input is a different input.
#[test]
fn every_single_byte_change_misses() {
    for len in [20usize, 32, 64] {
        let mut cache = KeccakCache::new();
        let a = pattern(len, 7);
        assert!(lookup(&mut cache, &a));
        for at in 0..len {
            for x in [0x01u8, 0x80] {
                let mut b = a.clone();
                b[at] ^= x;
                lookup(&mut cache, &b);
                lookup(&mut cache, &a);
            }
        }
    }
}

#[test]
fn many_inputs_stay_correct_past_capacity() {
    let mut cache = KeccakCache::new();
    for round in 0..2 {
        for seed in 0..20_000u64 {
            for len in [20usize, 32, 64] {
                let mut input = pattern(len, seed);
                // small big-endian integers, like storage slot numbers
                if seed % 3 == 0 {
                    input.fill(0);
                    input[len - 8..].copy_from_slice(&seed.to_be_bytes());
                }
                lookup(&mut cache, &input);
            }
        }
        let _ = round;
    }
}
