//! ★ The reproducible nonce search, and the reason it had to exist.
//!
//! `generate_nonce` returns *a* valid nonce. Which one is not a contract, and
//! under `parallel` it is rayon's `find_any` — whichever worker got there
//! first. That would be harmless if the nonce were merely recorded, but it is
//! **absorbed into the transcript**, so every challenge drawn after the first
//! grind depends on it. Two honest runs of the same prover therefore produce
//! different Merkle roots, different out-of-domain values and different
//! openings, and no normalisation of the nonce fields can undo that, because
//! the divergence is not in those fields.
//!
//! [`generate_nonce_smallest`] is the fix a byte gate needs: the smallest valid
//! nonce is a function of the seed and the factor alone. These tests pin the
//! three things that makes it worth anything — it is reproducible, it really is
//! the smallest, and it is valid — and they are written so each can fail.

use digest::Digest;

use crate::grinding::{generate_nonce, generate_nonce_smallest, is_valid_nonce};
use crate::hash::platform_keccak::PlatformKeccak256 as Keccak;

/// A factor small enough that the exhaustive scan below is instant and large
/// enough that the answer is not zero on most seeds.
const FACTOR: u8 = 12;

fn seed(tag: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = tag;
    for (i, b) in out.iter_mut().enumerate().skip(1) {
        *b = (i as u8).wrapping_mul(37).wrapping_add(tag);
    }
    out
}

/// It is reproducible: the same inputs give the same nonce.
#[test]
fn the_smallest_nonce_is_reproducible() {
    for tag in 0..4u8 {
        let s = seed(tag);
        let a = generate_nonce_smallest::<Keccak>(&s, FACTOR).expect("a nonce exists");
        let b = generate_nonce_smallest::<Keccak>(&s, FACTOR).expect("a nonce exists");
        assert_eq!(a, b, "seed {tag}: the smallest nonce must not vary");
    }
}

/// ★ It really is the smallest — checked against an exhaustive scan, which is a
/// different algorithm from the one under test.
///
/// This is the assertion that can fail if `find_first` is ever swapped back to
/// `find_any` for speed, which is exactly the regression the deterministic knob
/// exists to prevent.
#[test]
fn it_is_the_smallest_valid_nonce_and_not_merely_a_valid_one() {
    // The exhaustive loop below is vacuous when the answer is zero, so at least
    // one seed has to land above it for the test to be testing anything.
    let mut scanned = 0u64;
    for tag in 0..4u8 {
        let s = seed(tag);
        let n = generate_nonce_smallest::<Keccak>(&s, FACTOR).expect("a nonce exists");
        scanned += n;

        assert!(
            is_valid_nonce::<Keccak>(&s, n, FACTOR),
            "seed {tag}: the chosen nonce {n} does not pass the verifier's own check"
        );
        assert!(
            (0..n).all(|candidate| !is_valid_nonce::<Keccak>(&s, candidate, FACTOR)),
            "seed {tag}: a nonce below {n} is also valid, so {n} is not the smallest"
        );
    }
    assert!(
        scanned > 0,
        "every seed's smallest nonce was zero, so the minimality scan never ran"
    );
}

/// The unpinned search is still correct — it just promises less.
///
/// Stated as "valid, and never below the smallest" rather than "different":
/// asserting a difference would be a coin flip, and a test that fails at random
/// teaches nothing.
#[test]
fn the_unpinned_search_returns_a_valid_nonce_no_smaller_than_the_smallest() {
    for tag in 0..4u8 {
        let s = seed(tag);
        let smallest = generate_nonce_smallest::<Keccak>(&s, FACTOR).expect("a nonce exists");
        let any = generate_nonce::<Keccak>(&s, FACTOR).expect("a nonce exists");

        assert!(
            is_valid_nonce::<Keccak>(&s, any, FACTOR),
            "seed {tag}: the unpinned search returned an invalid nonce"
        );
        assert!(
            any >= smallest,
            "seed {tag}: {any} is below the exhaustively-checked smallest {smallest}"
        );
    }
}

/// The construction itself, against the spec in the module doc: the outer hash
/// of `inner ‖ nonce` must have `FACTOR` leading zero bits.
///
/// An independent reading of the same predicate — `is_valid_nonce` compares a
/// big-endian `u64` against a limit; this counts the bits — so the two cannot
/// be one transcription of the other.
#[test]
fn a_valid_nonce_really_does_have_the_leading_zeros() {
    let s = seed(1);
    let n = generate_nonce_smallest::<Keccak>(&s, FACTOR).expect("a nonce exists");

    // Rebuild the inner hash the way the module documents it.
    const PREFIX: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed];
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&PREFIX);
    inner_data[8..40].copy_from_slice(&s);
    inner_data[40] = FACTOR;
    let inner = Keccak::digest(inner_data);

    let mut outer_data = [0u8; 40];
    outer_data[..32].copy_from_slice(&inner);
    outer_data[32..].copy_from_slice(&n.to_be_bytes());
    let outer = Keccak::digest(outer_data);

    let leading = u64::from_be_bytes(outer[..8].try_into().unwrap()).leading_zeros();
    assert!(
        leading >= FACTOR as u32,
        "nonce {n} gives only {leading} leading zero bits, needed {FACTOR}"
    );
}
