//! The precomputed tree's staged download (`LAMBDA_VM_TREE_DOWNLOAD=staged`)
//! must hand the host the same nodes as the pageable download, byte for byte,
//! across every chunk boundary and a ragged tail (a full node buffer is
//! `2^k − 32` bytes), in both forms of the preprocessed commit (the legacy
//! row-major one and the column-major engine's), and leave the multiplicity
//! tree, the host LDE and the column-major handle as they were.
//!
//! Device tests: they need the card, like every other file in this directory.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math_cuda::DeviceHash;
use math_cuda::device::{
    STAGED_CHUNK_BYTES, backend, dtoh_staged_uncounted, set_tree_download_override,
    tree_download_totals,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

/// Nodes per staging buffer.
const NODE_CHUNK: usize = STAGED_CHUNK_BYTES / 32;

const COSET_OFFSET: u64 = 7;

fn random(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen::<u64>()).collect()
}

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

/// The first differing node, so a failure names the chunk it is in.
fn assert_same_nodes(got: &[[u8; 32]], want: &[[u8; 32]], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: node count");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: first difference at node {i} (chunk {}, node {} in it)",
            i / NODE_CHUNK,
            i % NODE_CHUNK
        );
    }
}

fn assert_same_u64(got: &[u64], want: &[u64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: first difference at u64 {i}");
    }
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

/// 32-byte elements through the generic staged download: every node exact,
/// the closure run once, at sizes around one chunk and past three.
#[test]
fn staged_node_download_is_exact_across_chunk_boundaries() {
    use cudarc::driver::DevicePtr;
    for (n, seed) in [
        (1usize, 1u64),
        (NODE_CHUNK - 1, 2),
        (NODE_CHUNK, 3),
        (NODE_CHUNK + 1, 4),
        // A full node buffer's shape: three buffers and a tail one node short.
        (4 * NODE_CHUNK - 1, 5),
    ] {
        let bytes: Vec<u8> = random(n * 4, seed)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let want: Vec<[u8; 32]> = bytes
            .chunks_exact(32)
            .map(|c| c.try_into().expect("32 bytes"))
            .collect();
        let be = backend().expect("cuda backend");
        let stream = be.next_stream();
        let dev = stream.clone_htod(&bytes).expect("htod");
        let ptr = {
            let (p, _record) = dev.device_ptr(&stream);
            p
        };
        let mut got: Vec<[u8; 32]> = Vec::new();
        let mut calls = 0usize;
        // SAFETY: `ptr` is `dev`, `n` nodes written by the upload queued on
        // `stream` above, written by nothing after, alive until the loop's end.
        unsafe {
            dtoh_staged_uncounted(&stream, ptr, n, &mut got, || {
                calls += 1;
                Ok(())
            })
        }
        .expect("staged node download");
        drop(dev);
        assert_eq!(calls, 1, "n={n}: the closure must run exactly once");
        assert_same_nodes(&got, &want, &format!("nodes n={n}"));
    }
}

/// What one preprocessed commit hands back, for comparing the two downloads.
struct Commit {
    pre: Option<Vec<[u8; 32]>>,
    mult_root: [u8; 32],
    host_lde: Vec<u64>,
    handle: Vec<u64>,
}

#[allow(clippy::too_many_arguments)]
fn commit(
    staged: bool,
    engine: bool,
    row_major: &[u64],
    n: usize,
    cols: usize,
    blowup: usize,
    split: usize,
    build_precomputed: bool,
    retain_host_lde: bool,
    rows_per_leaf: usize,
) -> Commit {
    let weights = coset_weights_u64(n, COSET_OFFSET);
    set_tree_download_override(Some(staged));
    let (pre, handle, host_lde) = math_cuda::lde_cm::with_engine(engine, || {
        math_cuda::lde::coset_lde_row_major_split_trees_rpl(
            row_major,
            None,
            DeviceHash::Rpx256,
            n,
            cols,
            blowup,
            &weights,
            split,
            build_precomputed,
            retain_host_lde,
            rows_per_leaf,
        )
    })
    .expect("split-tree commit");
    set_tree_download_override(None);
    Commit {
        pre,
        mult_root: handle.tree.as_ref().expect("the multiplicity tree").root,
        host_lde,
        handle: download_handle(handle.buf.as_ref(), handle.ready.as_deref()),
    }
}

/// Both downloads over one commit, in both forms, with and without the
/// precomputed tree: the same nodes, multiplicity root, host LDE and handle;
/// each arm's counter moved by exactly its tree; no counter moved on a cache hit.
#[test]
fn the_staged_tree_download_is_the_pageable_download() {
    // (log_n, blowup, cols, split, rows per leaf, engine, retain host LDE)
    let cases: &[(u32, usize, usize, usize, usize, bool, bool)] = &[
        // Legacy row-major form (a kept host LDE takes it): 2^21 row-pair
        // leaves, a 128 MiB − 32 B tree = three chunks and a ragged tail.
        (20, 4, 3, 1, 2, false, true),
        // The column-major engine: 2^22 one-row leaves, 256 MiB − 32 B.
        (20, 4, 3, 2, 1, true, false),
        // One chunk, both forms.
        (12, 2, 4, 2, 2, false, true),
        (12, 2, 4, 2, 1, true, false),
    ];
    for (i, &(log_n, blowup, cols, split, rpl, engine, retain)) in cases.iter().enumerate() {
        let n = 1usize << log_n;
        if engine {
            assert!(
                math_cuda::lde_cm::supports(n, blowup),
                "log_n={log_n} blowup={blowup}: the engine must take this shape"
            );
        }
        let row_major = random(n * cols, 0x7D0 + i as u64);
        let tree_bytes = (2 * (n * blowup / rpl) - 1) * 32;
        for build in [true, false] {
            let what = format!(
                "log_n={log_n} blowup={blowup} cols={cols} split={split} rpl={rpl} \
                 engine={engine} retain={retain} build_precomputed={build}"
            );
            let t0 = tree_download_totals();
            let pageable = commit(
                false, engine, &row_major, n, cols, blowup, split, build, retain, rpl,
            );
            let t1 = tree_download_totals();
            let staged = commit(
                true, engine, &row_major, n, cols, blowup, split, build, retain, rpl,
            );
            let t2 = tree_download_totals();

            assert_eq!(
                pageable.pre.is_some(),
                build,
                "{what}: pageable tree iff requested"
            );
            assert_eq!(
                staged.pre.is_some(),
                build,
                "{what}: staged tree iff requested"
            );
            if let (Some(p), Some(s)) = (&pageable.pre, &staged.pre) {
                assert_eq!(p.len() * 32, tree_bytes, "{what}: full node buffer");
                assert_same_nodes(s, p, &format!("{what}: precomputed nodes"));
            }
            assert_eq!(
                pageable.mult_root, staged.mult_root,
                "{what}: multiplicity root"
            );
            assert_same_u64(
                &staged.host_lde,
                &pageable.host_lde,
                &format!("{what}: host LDE"),
            );
            assert_same_u64(&staged.handle, &pageable.handle, &format!("{what}: handle"));

            let (dp, ds) = (t1.since(&t0), t2.since(&t1));
            let one = u64::from(build);
            assert_eq!(dp.trees, [one, 0], "{what}: the pageable arm's counter");
            assert_eq!(
                dp.bytes,
                [one * tree_bytes as u64, 0],
                "{what}: pageable bytes"
            );
            assert_eq!(ds.trees, [0, one], "{what}: the staged arm's counter");
            assert_eq!(
                ds.bytes,
                [0, one * tree_bytes as u64],
                "{what}: staged bytes"
            );
        }
    }
}
