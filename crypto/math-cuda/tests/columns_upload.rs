//! The staged column upload leaves the card holding exactly the columns.
//!
//! `DeviceColumns::upload` under `LAMBDA_VM_TRACE_UPLOAD` stages each column
//! through a pair of its own from several threads and zeroes its all-zero tail
//! on the card instead of sending it. Read back, the store must equal the host
//! columns laid end to end, value for value, and equal the pageable upload's.
//!
//! The buffer comes from a pool that just held `u64::MAX` everywhere, so a tail
//! the upload forgot to zero reads back as garbage rather than as the zeros a
//! fresh allocation might happen to hold.
//!
//! Needs a card: `cargo test --release -p math-cuda --test columns_upload --
//! --ignored --test-threads=1`.

use math_cuda::columns::{
    DeviceColumns, ZERO_TAIL_MIN_BYTES, sent_len, set_trace_upload_override, upload_totals,
};
use math_cuda::device::{STAGED_CHUNK_BYTES, backend};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Columns with every tail shape the upload distinguishes: none, shorter than
/// the cut, exactly the cut, long, all zero, a nonzero value deep in a zero run,
/// and one column longer than a staging chunk with a tail across chunks.
fn columns() -> Vec<Vec<u64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(0x7ace);
    let min = ZERO_TAIL_MIN_BYTES / 8;
    let chunk = STAGED_CHUNK_BYTES / 8;
    let mut filled = |len: usize, live: usize| -> Vec<u64> {
        let mut c = vec![0u64; len];
        for v in &mut c[..live] {
            *v = rng.r#gen::<u64>() | 1;
        }
        c
    };
    let mut out = vec![
        filled(1 << 16, 1 << 16),
        filled(1 << 16, (1 << 16) - (min - 1)),
        filled(1 << 16, (1 << 16) - min),
        filled(1 << 16, 1_000),
        filled(1 << 16, 0),
        filled(3, 3),
        filled(chunk + (chunk / 2), chunk / 3),
    ];
    let mut holes = filled(1 << 17, 10);
    holes[(1 << 17) - min - 7] = 42;
    out.push(holes);
    // Many small tables' columns, more than the threads.
    for k in 0..40 {
        out.push(filled(1 << 12, (k * 97) % (1 << 12)));
    }
    out
}

/// Fill the pool with `u64::MAX` and give it back, so the next allocation of
/// that size is dirty.
fn dirty_the_pool(elems: usize) {
    let be = backend().expect("a card");
    let stream = be.next_stream();
    let mut junk = stream.alloc_zeros::<u64>(elems).expect("alloc");
    let ones = vec![u64::MAX; elems];
    stream.memcpy_htod(&ones, &mut junk).expect("fill");
    stream.synchronize().expect("sync");
    drop(junk);
    stream.synchronize().expect("sync");
}

fn read_back(store: &DeviceColumns) -> Vec<u64> {
    let stream = store.stream();
    let back = stream.clone_dtoh(&*store.buffer()).expect("dtoh");
    stream.synchronize().expect("sync");
    back
}

#[test]
#[ignore = "needs a card"]
fn the_staged_upload_holds_the_columns_exactly() {
    let host = columns();
    let refs: Vec<&[u64]> = host.iter().map(Vec::as_slice).collect();
    let expected: Vec<u64> = host.iter().flatten().copied().collect();
    let expected_skip: u64 = host
        .iter()
        .map(|c| ((c.len() - sent_len(c)) * 8) as u64)
        .sum();
    assert!(expected_skip > 0, "the fixture has no tail to cut");

    dirty_the_pool(expected.len());
    set_trace_upload_override(Some(false));
    let pageable = DeviceColumns::upload(&refs).expect("pageable upload");
    let plain = read_back(&pageable);
    assert_eq!(pageable.upload_record().threads, 0);
    assert_eq!(pageable.upload_record().skipped, 0);
    drop(pageable);

    dirty_the_pool(expected.len());
    let before = upload_totals();
    set_trace_upload_override(Some(true));
    let staged = DeviceColumns::upload(&refs).expect("staged upload");
    set_trace_upload_override(None);
    let after = upload_totals();
    let record = staged.upload_record();
    assert!(record.threads >= 1, "the staged arm ran pageable");
    assert_eq!(after.1, before.1 + 1);
    assert_eq!(record.skipped, expected_skip, "zero tails left behind");
    assert_eq!(record.bytes, expected.len() as u64 * 8);
    let got = read_back(&staged);

    assert_eq!(plain.len(), expected.len());
    assert_eq!(got.len(), expected.len());
    let first_bad = |a: &[u64]| a.iter().zip(&expected).position(|(x, y)| x != y);
    assert_eq!(first_bad(&plain), None, "CONTROL: the pageable upload");
    assert_eq!(first_bad(&got), None, "the staged upload");
    for (k, column) in refs.iter().enumerate() {
        let (at, len) = staged.span(k);
        assert_eq!(&got[at..at + len], *column, "column {k}");
    }
}

/// The dirty pool is what makes the tail check bite: an upload that skipped the
/// tail and did not zero it would read `u64::MAX` there. Shown directly: a
/// dirtied allocation of the same size, left unwritten, does not read as zeros.
#[test]
#[ignore = "needs a card"]
fn a_dirty_pool_hands_back_garbage() {
    let elems = 1 << 20;
    dirty_the_pool(elems);
    let be = backend().expect("a card");
    let stream = be.next_stream();
    // SAFETY: read back only to show what an unwritten buffer holds.
    let buf = unsafe { math_cuda::device::alloc_or_trim::<u64>(&stream, elems) }.expect("alloc");
    let back = stream.clone_dtoh(&buf).expect("dtoh");
    stream.synchronize().expect("sync");
    assert!(
        back.iter().any(|&v| v != 0),
        "CONTROL: the pool handed back zeros, so the tail check could not fail"
    );
}
