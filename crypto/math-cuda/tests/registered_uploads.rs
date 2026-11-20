//! An upload from a buffer page-locked in place (`math_cuda::registered`) goes
//! to the card as direct DMAs and must move every byte exactly as the staged
//! path does: misaligned heads and tails, sizes across several pieces, a source
//! inside one piece, and a source with a pageable hole (which must fall back to
//! the staged path whole). Unregistering leaves the next upload staged.
//!
//! Device tests: they need the card, like every other file in this directory.
//! Each test takes one lock: the registrations and counters are process-wide.

use std::sync::Mutex;

use math_cuda::device::{backend, htod_staged};
use math_cuda::registered::{self, PAGE_BYTES, PIECE_BYTES};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

static LOCK: Mutex<()> = Mutex::new(());

fn random(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen::<u64>()).collect()
}

/// Upload `src` through `htod_staged`, read it back plainly; with the direct
/// calls the upload made.
fn roundtrip(src: &[u64]) -> (Vec<u64>, u64) {
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let mut dst = stream.alloc_zeros::<u64>(src.len()).expect("device alloc");
    let before = registered::stats().direct_calls;
    htod_staged(&stream, src, &mut dst.slice_mut(0..src.len())).expect("htod_staged");
    let calls = registered::stats().direct_calls - before;
    let back = stream.clone_dtoh(&dst).expect("dtoh");
    stream.synchronize().expect("sync");
    (back, calls)
}

fn first_difference(got: &[u64], want: &[u64]) -> Option<usize> {
    got.iter().zip(want).position(|(a, b)| a != b)
}

#[test]
fn a_registered_source_uploads_directly_and_byte_for_byte() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // 2.5 pieces plus a few words, so the head, the tail and a piece edge are
    // all inside the upload.
    let n = (5 * PIECE_BYTES / 2) / 8 + 13;
    let buf = random(n, 7);
    let r = registered::register(buf.as_ptr().cast(), n * 8);
    assert!(r.bytes > 0 && r.refused == 0, "the driver locked it: {r:?}");
    for (lo, hi) in [
        (0, n),
        (3, n - 5),
        (PAGE_BYTES / 8 + 1, PIECE_BYTES / 8 + 77),
    ] {
        let (back, calls) = roundtrip(&buf[lo..hi]);
        assert_eq!(first_difference(&back, &buf[lo..hi]), None, "[{lo}, {hi})");
        assert_eq!(calls, 1, "[{lo}, {hi}) went direct");
    }
    registered::unregister(r);
    let (back, calls) = roundtrip(&buf);
    assert_eq!(first_difference(&back, &buf), None);
    assert_eq!(calls, 0, "unregistered: staged again");
}

#[test]
fn a_source_with_a_pageable_piece_is_staged_whole() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = 3 * PIECE_BYTES / 8;
    let a = random(n, 11);
    // Two separate registrations with the middle piece's worth left out.
    let head = registered::register(a.as_ptr().cast(), PIECE_BYTES);
    // SAFETY: inside `a`.
    let far = unsafe { a.as_ptr().add(2 * PIECE_BYTES / 8) };
    let tail = registered::register(far.cast(), PIECE_BYTES);
    let (back, calls) = roundtrip(&a);
    assert_eq!(first_difference(&back, &a), None);
    assert_eq!(calls, 0, "a pageable hole: the whole upload staged");
    registered::unregister(head);
    registered::unregister(tail);
}
