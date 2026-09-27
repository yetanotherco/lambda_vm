//! The per-transfer, double-buffered pinned staging (`htod_staged`,
//! `dtoh_staged_into`) must move every byte exactly, across every chunk
//! boundary, from any number of threads at once — and the fused commits, the
//! row-major ones and the column-major engine's, must produce the SAME roots,
//! the same host LDE and the same column-major handle through it as through the
//! shared slab (`LAMBDA_VM_STAGING_SHARED_SLAB=1`).
//!
//! Device tests: they need the card, like every other file in this directory.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math_cuda::DeviceHash;
use math_cuda::device::{
    STAGED_CHUNK_BYTES, backend, dtoh_staged_into, htod_staged, set_staging_pairs_override,
    staging_totals,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

/// u64s per staging buffer.
const CHUNK: usize = STAGED_CHUNK_BYTES / 8;

fn random(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen::<u64>()).collect()
}

/// The first differing offset, so a failure names the chunk it is in.
fn assert_same(got: &[u64], want: &[u64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: first difference at u64 {i} (chunk {}, offset {} in it): {:#x} != {:#x}",
            i / CHUNK,
            i % CHUNK,
            got[i],
            want[i]
        );
    }
}

/// Upload through `htod_staged`, read back plainly.
fn upload_roundtrip(src: &[u64]) -> Vec<u64> {
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let mut dst = stream.alloc_zeros::<u64>(src.len()).expect("device alloc");
    htod_staged(&stream, src, &mut dst.slice_mut(0..src.len())).expect("htod_staged");
    let back = stream.clone_dtoh(&dst).expect("dtoh");
    stream.synchronize().expect("sync");
    back
}

/// Upload plainly, read back through `dtoh_staged_into`; returns the values
/// and how many times the closure ran.
fn download_roundtrip(src: &[u64]) -> (Vec<u64>, usize) {
    use cudarc::driver::DevicePtr;
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let dev = stream.clone_htod(src).expect("htod");
    let ptr = {
        let (p, _record) = dev.device_ptr(&stream);
        p
    };
    let mut out = Vec::new();
    let mut calls = 0usize;
    // SAFETY: `ptr` is `dev`, written by the upload queued on `stream` above,
    // written by nothing after, and alive until this function returns.
    let tag = unsafe {
        dtoh_staged_into(&stream, ptr, src.len(), &mut out, || {
            calls += 1;
            Ok(0xC0FFEEu32)
        })
    }
    .expect("dtoh_staged_into");
    assert_eq!(tag, 0xC0FFEE, "the closure's value is returned");
    drop(dev);
    (out, calls)
}

#[test]
fn staged_upload_is_exact_across_chunk_boundaries() {
    for (n, seed) in [
        (1usize, 1u64),
        (CHUNK - 1, 2),
        (CHUNK, 3),
        (CHUNK + 1, 4),
        // Three full buffers and a partial tail: both buffers reused.
        (3 * CHUNK + 12_345, 5),
    ] {
        let src = random(n, seed);
        assert_same(&upload_roundtrip(&src), &src, &format!("upload n={n}"));
    }
}

#[test]
fn staged_download_is_exact_and_runs_the_closure_once() {
    for (n, seed) in [
        (1usize, 11u64),
        (CHUNK - 1, 12),
        (CHUNK, 13),
        (CHUNK + 1, 14),
        (2 * CHUNK, 15),
        (3 * CHUNK + 777, 16),
    ] {
        let src = random(n, seed);
        let (got, calls) = download_roundtrip(&src);
        assert_eq!(calls, 1, "n={n}: the closure must run exactly once");
        assert_same(&got, &src, &format!("download n={n}"));
    }
    // Zero values: the closure still runs once and nothing lands.
    let (got, calls) = download_roundtrip(&[]);
    assert!(got.is_empty());
    assert_eq!(calls, 1, "n=0: the closure must still run once");
}

/// ⛔ THE HAZARD THE EVENTS EXIST FOR. `htod_staged` returns without waiting
/// for its last DMA, so the SAME thread's next transfer can borrow the same
/// pair while that DMA still reads a buffer. A download into that buffer must
/// wait for it; if it did not, the upload would arrive with the download's
/// bytes in it. Interleaved back to back on different streams, many times.
#[test]
fn a_download_never_overwrites_a_buffer_an_upload_is_still_reading() {
    let be = backend().expect("cuda backend");
    for round in 0..8u64 {
        let up = random(2 * CHUNK + 99, 100 + round);
        let down = random(2 * CHUNK + 55, 200 + round);
        let s_up = be.next_stream();
        let mut up_dev = s_up.alloc_zeros::<u64>(up.len()).expect("alloc");
        htod_staged(&s_up, &up, &mut up_dev.slice_mut(0..up.len())).expect("htod_staged");
        // No sync: the upload's DMAs may still be reading the pair.
        let (got, _) = download_roundtrip(&down);
        assert_same(&got, &down, &format!("round {round}: download"));
        let back = s_up.clone_dtoh(&up_dev).expect("dtoh");
        s_up.synchronize().expect("sync");
        assert_same(&back, &up, &format!("round {round}: upload"));
    }
}

/// More threads than pairs: every transfer exact, and the pool never grows
/// past its cap (the extra threads wait for a pair instead).
#[test]
fn concurrent_transfers_beyond_the_pool_cap_are_exact() {
    const THREADS: usize = 12;
    std::thread::scope(|s| {
        for t in 0..THREADS {
            s.spawn(move || {
                for round in 0..3u64 {
                    let n = CHUNK + 1000 * (t + 1) + round as usize;
                    let seed = 1_000 + (t as u64) * 10 + round;
                    let src = random(n, seed);
                    assert_same(
                        &upload_roundtrip(&src),
                        &src,
                        &format!("thread {t} round {round}: upload"),
                    );
                    let (got, _) = download_roundtrip(&src);
                    assert_same(&got, &src, &format!("thread {t} round {round}: download"));
                }
            });
        }
    });
    let totals = staging_totals();
    assert!(
        totals.pairs <= math_cuda::device::MAX_STAGING_PAIRS,
        "the pool grew past its cap: {totals:?}"
    );
}

// ── The fused commits, one arm against the other ─────────────────────────────

const COSET_OFFSET: u64 = 7;

/// `weights[i] = g^i / n` — the `LdeTwiddles::coset_weights` format.
fn coset_weights_u64(n: usize, g: u64) -> Vec<u64> {
    let inv_n = Fp::from(n as u64).inv().unwrap();
    let g_fp = Fp::from_raw(g);
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(*cur.value());
        cur = &cur * &g_fp;
    }
    w
}

/// D2H a handle's column-major buffer once its `ready` event has fired.
fn download_handle(
    buf: &math_cuda::CudaSlice<u64>,
    ready: Option<&math_cuda::device::PooledEvent>,
) -> Vec<u64> {
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    if let Some(ev) = ready {
        stream.wait(ev.event()).expect("wait on the ready event");
    }
    let v = stream.clone_dtoh(buf).expect("D2H of the handle");
    stream.synchronize().expect("sync");
    v
}

/// Runs `f` through the shared slab, then through pairs, on this thread.
fn both_paths<T>(f: impl Fn() -> T) -> (T, T) {
    set_staging_pairs_override(Some(false));
    let shared = f();
    set_staging_pairs_override(Some(true));
    let pairs = f();
    set_staging_pairs_override(None);
    (shared, pairs)
}

/// Shapes whose host input clears the 8 MB staged-upload floor and whose LDE
/// spans one buffer, exactly two, or two plus a tail.
const SHAPES: &[(usize, usize, usize)] = &[
    // (log_n, blowup, base columns)
    (18, 2, 5),  // 10.5 MB in, 21 MB LDE: one chunk
    (17, 4, 9),  // 9.4 MB in, 37.7 MB LDE: a chunk and a tail
    (16, 4, 32), // 16.8 MB in, 67 MB LDE: exactly two chunks
];

#[test]
fn the_base_commit_is_identical_through_either_staging() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let n = 1usize << log_n;
        let row_major = random(n * cols, 0xB5E0 + i as u64);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let before = staging_totals();
        let (off, on) = both_paths(|| {
            let (handle, host) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep_rpl(
                &row_major,
                None,
                DeviceHash::Rpx256,
                n,
                cols,
                blowup,
                &weights,
                true,
                2,
            )
            .expect("fused base commit");
            let dev = download_handle(handle.buf.as_ref(), handle.ready.as_deref());
            let root = handle.tree.as_ref().expect("keep path keeps the tree").root;
            (root, host, dev)
        });
        let what = format!("base log_n={log_n} blowup={blowup} cols={cols}");
        assert_eq!(off.0, on.0, "{what}: root");
        assert_same(&on.1, &off.1, &format!("{what}: host LDE"));
        assert_same(&on.2, &off.2, &format!("{what}: column-major handle"));
        // ⛔ The check that can fail if the override stopped routing: the staged
        // path must have moved this shape's bytes.
        let after = staging_totals();
        assert!(
            after.staged_in_bytes >= before.staged_in_bytes + (n * cols * 8) as u64,
            "{what}: the upload did not go through the staged path: {before:?} -> {after:?}"
        );
        assert!(
            after.staged_out_bytes >= before.staged_out_bytes + (n * blowup * cols * 8) as u64,
            "{what}: the download did not go through the staged path: {before:?} -> {after:?}"
        );
    }
}

#[test]
fn the_ext3_commit_is_identical_through_either_staging() {
    // `m` ext3 columns = `3 m` base columns row-major.
    for (i, &(log_n, blowup, m)) in [(16usize, 4usize, 6usize), (17, 2, 3)].iter().enumerate() {
        let n = 1usize << log_n;
        let row_major = random(n * m * 3, 0xE530 + i as u64);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let (off, on) = both_paths(|| {
            let (handle, host) =
                math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep_rpl(
                    &row_major,
                    DeviceHash::Rpx256,
                    n,
                    m,
                    blowup,
                    &weights,
                    true,
                    2,
                )
                .expect("fused ext3 commit");
            let dev = download_handle(handle.buf.as_ref(), handle.ready.as_deref());
            let root = handle.tree.as_ref().expect("keep path keeps the tree").root;
            (root, host, dev)
        });
        let what = format!("ext3 log_n={log_n} blowup={blowup} m={m}");
        assert_eq!(off.0, on.0, "{what}: root");
        assert_same(&on.1, &off.1, &format!("{what}: host LDE"));
        assert_same(&on.2, &off.2, &format!("{what}: column-major handle"));
    }
}

#[test]
fn the_split_tree_commit_is_identical_through_either_staging() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let n = 1usize << log_n;
        let split_col = (cols / 2).max(1);
        let row_major = random(n * cols, 0x5B17 + i as u64);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let (off, on) = both_paths(|| {
            let (pre, handle, host) = math_cuda::lde::coset_lde_row_major_split_trees_rpl(
                &row_major,
                None,
                DeviceHash::Rpx256,
                n,
                cols,
                blowup,
                &weights,
                split_col,
                true,
                true,
                2,
            )
            .expect("fused split-tree commit");
            let dev = download_handle(handle.buf.as_ref(), handle.ready.as_deref());
            let mult_root = handle.tree.as_ref().expect("the mult tree").root;
            (pre, mult_root, host, dev)
        });
        let what = format!("split log_n={log_n} blowup={blowup} cols={cols}");
        assert_eq!(off.0, on.0, "{what}: precomputed tree nodes");
        assert_eq!(off.1, on.1, "{what}: multiplicity root");
        assert_same(&on.2, &off.2, &format!("{what}: host LDE"));
        assert_same(&on.3, &off.3, &format!("{what}: column-major handle"));
    }
}

/// The column-major engine's device-only commit (the default for the shapes it
/// takes) stages its upload the same way: the same root and column-major handle
/// through either staging, no host LDE, and the upload's bytes moved staged.
#[test]
fn the_engine_commit_is_identical_through_either_staging() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let n = 1usize << log_n;
        assert!(
            math_cuda::lde_cm::supports(n, blowup),
            "log_n={log_n} blowup={blowup}: the engine must take this shape"
        );
        let row_major = random(n * cols, 0xC011 + i as u64);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let before = staging_totals();
        let (off, on) = both_paths(|| {
            math_cuda::lde_cm::with_engine(true, || {
                let (handle, host) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep_rpl(
                    &row_major,
                    None,
                    DeviceHash::Rpx256,
                    n,
                    cols,
                    blowup,
                    &weights,
                    false,
                    2,
                )
                .expect("device-only base commit");
                assert!(host.is_empty(), "a device-only commit returns no host LDE");
                let dev = download_handle(handle.buf.as_ref(), handle.ready.as_deref());
                let root = handle.tree.as_ref().expect("keep path keeps the tree").root;
                (root, dev)
            })
        });
        let what = format!("engine log_n={log_n} blowup={blowup} cols={cols}");
        assert_eq!(off.0, on.0, "{what}: root");
        assert_same(&on.1, &off.1, &format!("{what}: column-major handle"));
        let after = staging_totals();
        assert!(
            after.staged_in_bytes >= before.staged_in_bytes + (n * cols * 8) as u64,
            "{what}: the upload did not go through the staged path: {before:?} -> {after:?}"
        );
    }
}
