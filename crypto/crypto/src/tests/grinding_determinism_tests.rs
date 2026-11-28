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

/// ★ The chunked search finds the nonce the whole-range search found: the same
/// smallest nonce, so a byte gate's proof is the bytes it was before the chunks.
/// Factors up to 16 and twelve seeds a factor, against the search it replaced.
#[cfg(feature = "parallel")]
#[test]
fn the_chunked_smallest_nonce_is_the_whole_range_ones() {
    use crate::grinding::generate_nonce_smallest_whole_range;
    let mut above_a_chunk = 0;
    let chunk = crate::grinding::SMALLEST_CHUNK_PER_THREAD * rayon::current_num_threads() as u64;
    for factor in [1u8, 4, 8, 12, 14, 16] {
        for tag in 0..12u8 {
            let s = seed(tag.wrapping_mul(17).wrapping_add(factor));
            let chunked = generate_nonce_smallest::<Keccak>(&s, factor).expect("a nonce exists");
            let whole =
                generate_nonce_smallest_whole_range::<Keccak>(&s, factor).expect("a nonce exists");
            assert_eq!(chunked, whole, "factor {factor}, seed {tag}");
            above_a_chunk += usize::from(chunked >= chunk);
        }
    }
    // The chunks' own loop is exercised only past the first chunk: some seed at
    // 16 bits must land beyond it on any pool of at most 16 threads.
    if rayon::current_num_threads() <= 16 {
        assert!(above_a_chunk > 0, "every answer fell in the first chunk");
    }
}

/// The chunk loop at its edges, on a predicate the test controls: the answer
/// at a chunk's first and last nonce, one past a chunk, deep in a later chunk,
/// and the smallest of several valid nonces in one chunk.
#[cfg(feature = "parallel")]
#[test]
fn the_chunk_loop_returns_the_smallest_at_every_edge() {
    use crate::grinding::smallest_in_chunks;
    let per = 8u64;
    let chunk = per * rayon::current_num_threads() as u64;
    for want in [
        0,
        1,
        chunk - 1,
        chunk,
        chunk + 1,
        2 * chunk - 1,
        7 * chunk + 3,
    ] {
        assert_eq!(
            smallest_in_chunks(0..=u64::MAX, per, |n| n == want),
            Some(want),
            "at {want}"
        );
        assert_eq!(
            smallest_in_chunks(0..=u64::MAX, per, |n| n >= want && (n - want) % 3 == 0),
            Some(want),
            "the smallest of several from {want}"
        );
    }
    // A chunk of one nonce a thread still walks every nonce in order.
    assert_eq!(
        smallest_in_chunks(0..=u64::MAX, 1, |n| n == 1000),
        Some(1000)
    );
}

#[cfg(feature = "parallel")]
#[test]
fn the_search_checks_the_last_partial_chunk_and_exhausts_the_range() {
    use crate::grinding::smallest_in_chunks;
    use std::sync::Mutex;

    let per = 4;
    let chunk = per * rayon::current_num_threads() as u64;
    let range = u64::MAX - 2 * chunk - 1..=u64::MAX;
    for want in [u64::MAX - 1, u64::MAX] {
        assert_eq!(
            smallest_in_chunks(range.clone(), per, |n| n == want),
            Some(want)
        );
    }
    let seen = Mutex::new(Vec::new());
    assert_eq!(
        smallest_in_chunks(range.clone(), per, |n| {
            seen.lock().unwrap().push(n);
            false
        }),
        None
    );
    let mut seen = seen.into_inner().unwrap();
    seen.sort_unstable();
    assert_eq!(seen, range.collect::<Vec<_>>());
}

#[cfg(feature = "parallel")]
#[test]
fn the_chunked_search_preserves_the_post_grind_transcript() {
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use crate::fiat_shamir::is_transcript::IsTranscript;
    use crate::grinding::generate_nonce_smallest_whole_range;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;

    type Transcript = DefaultTranscript<Degree3GoldilocksExtensionField>;
    for threads in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for tag in 0..4 {
                let mut chunked = Transcript::new(&seed(tag));
                chunked.sample_field_element();
                chunked.append_bytes(&seed(tag + 16));
                let mut whole = chunked.clone();
                let state = chunked.state();
                let nonce = generate_nonce_smallest::<Keccak>(&state, FACTOR).unwrap();
                let old_nonce =
                    generate_nonce_smallest_whole_range::<Keccak>(&state, FACTOR).unwrap();
                chunked.append_bytes(&nonce.to_be_bytes());
                whole.append_bytes(&old_nonce.to_be_bytes());
                assert_eq!(chunked.state(), whole.state(), "pool {threads}, seed {tag}");
                assert_eq!(
                    chunked.sample_field_element(),
                    whole.sample_field_element(),
                    "pool {threads}, seed {tag}"
                );
            }
        });
    }
}

/// Pace probe (not a gate): the chunked search against the whole-range one at
/// 18 and 20 bits on Keccak, on this machine's pool. Prints one line a factor.
/// `cargo test -p crypto --release --features parallel -- --ignored --nocapture
/// smallest_nonce_pace`.
#[cfg(feature = "parallel")]
#[test]
#[ignore = "a pace probe: prints seconds, asserts only equality"]
fn smallest_nonce_pace() {
    use crate::grinding::generate_nonce_smallest_whole_range;
    for factor in [18u8, 20] {
        let (mut chunked_s, mut whole_s, mut nonces) = (0.0, 0.0, 0u64);
        for tag in 0..4u8 {
            let s = seed(tag.wrapping_add(100));
            let t = std::time::Instant::now();
            let a = generate_nonce_smallest::<Keccak>(&s, factor).expect("a nonce");
            chunked_s += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let b = generate_nonce_smallest_whole_range::<Keccak>(&s, factor).expect("a nonce");
            whole_s += t.elapsed().as_secs_f64();
            assert_eq!(a, b, "factor {factor}, seed {tag}");
            nonces += a;
        }
        println!(
            "SMALLEST NONCE PACE: factor {factor} · threads {} · 4 seeds · nonces scanned {nonces} · \
             chunked {chunked_s:.3}s · whole range {whole_s:.3}s · ratio {:.1}",
            rayon::current_num_threads(),
            whole_s / chunked_s.max(1e-9)
        );
    }
}
