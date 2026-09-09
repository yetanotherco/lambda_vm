//! Round-trip coverage for `htod_via`'s chunk loop: uploads larger than the
//! 64 MB pinned-staging chunk must arrive intact across every chunk boundary
//! (a stale slab or a bad byte offset would corrupt exactly one chunk).

use math_cuda::device::{backend, htod_via};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn roundtrip(n_u64: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let src: Vec<u64> = (0..n_u64).map(|_| rng.r#gen::<u64>()).collect();

    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let mut dst = stream.alloc_zeros::<u64>(n_u64).expect("device alloc");
    htod_via(
        &stream,
        be.pinned_staging(),
        &be.ctx,
        &src,
        &mut dst.slice_mut(0..n_u64),
    )
    .expect("htod_via");

    let back = stream.clone_dtoh(&dst).expect("dtoh");
    stream.synchronize().expect("sync");
    assert_eq!(src.len(), back.len());
    // Compare in chunks so a failure names the offset instead of dumping 100M+ values.
    for (i, (a, b)) in src.iter().zip(back.iter()).enumerate() {
        assert_eq!(
            a, b,
            "htod_via round-trip mismatch at u64 offset {i} (n={n_u64})"
        );
    }
}

#[test]
fn htod_via_single_chunk_roundtrip() {
    // Below the 64 MB chunk: single iteration of the loop.
    roundtrip(1 << 20, 42);
}

#[test]
fn htod_via_multi_chunk_roundtrip() {
    // 3 full chunks + a partial tail: exercises slab reuse across iterations
    // and the final short chunk. 64 MB chunk = 2^23 u64s.
    roundtrip((3 << 23) + 12345, 43);
}
