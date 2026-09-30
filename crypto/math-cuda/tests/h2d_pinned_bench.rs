//! What the bus gives a column upload from pinned memory, with no host copy.
//!
//! D-TRACE's box request 1. Its stage 1b would have the producer write the
//! epoch's columns straight into pinned slots, so the prover's upload becomes
//! pure DMA. That is worth building only if a DMA from pinned memory beats the
//! pageable copy the upload does today by enough: the stop rule, pre-registered
//! in G6-LEDGER.md §8, is **pinned ≥ 1.3× pageable over an epoch's columns**.
//!
//! Measured here, one stream each unless named:
//! * pageable: `memcpy_htod` from a `Vec`, what `DeviceColumns::upload` does;
//! * pinned: `cuMemcpyHtoDAsync` from `cuMemHostAlloc` memory, nothing copied
//!   on the host;
//! * pinned over two streams, alternating, to see whether the link takes more
//!   than one stream gives.
//!
//! Each at fixed column sizes and over epoch 0's real column sizes (the G6
//! census, 2.06 GB in 393 columns). Every rate is bytes over wall time from the
//! first enqueue to the stream's synchronize, the median of five runs; GB = 1e9
//! bytes.
//!
//! Server-only; needs a real device:
//! `cargo test --release -p math-cuda --test h2d_pinned_bench -- --ignored --nocapture`.

use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use math_cuda::device::backend;

const RUNS: usize = 5;
/// Bytes moved per measurement at a fixed column size.
const PER_SIZE_BYTES: usize = 1 << 31;

/// Epoch 0's committed tables in the G6 census (`rows_by_table.tsv`, #0):
/// `(rows per column, columns)`, every instance its own entry.
const EPOCH0: &[(usize, usize)] = &[
    (1 << 20, 21), // BITWISE
    (1 << 20, 6),  // DECODE
    (32, 10),      // KECCAK_RC
    (128, 5),      // REGISTER
    (1 << 21, 38), // CPU
    (1 << 20, 17), // LT
    (1 << 18, 29), // SHIFT
    (1 << 17, 49), // MEMW
    (1 << 20, 29), // MEMW_A
    (1 << 19, 18), // LOAD
    (1 << 11, 26), // MUL
    (1 << 17, 14), // BRANCH
    (1 << 21, 10), // MEMW_R
    (1 << 21, 10), // MEMW_R
    (1 << 18, 10), // MEMW_R
    (1 << 13, 12), // EQ
    (1 << 17, 26), // BYTEWISE
    (1 << 19, 16), // STORE
    (1 << 10, 38), // CPU32
    (1 << 21, 9),  // L2G_MEMORY
];

/// Page-locked host memory from `cuMemHostAlloc`, freed on drop.
struct Pinned {
    ptr: *mut u8,
}

impl Pinned {
    fn new(bytes: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: a plain allocation of `bytes` page-locked bytes; the context
        // is bound by the caller.
        unsafe {
            cudarc::driver::sys::cuMemHostAlloc(&mut ptr, bytes, 0)
                .result()
                .expect("cuMemHostAlloc");
            // Touch every page so the first measured run pays no fault.
            std::ptr::write_bytes(ptr as *mut u8, 0x5a, bytes);
        }
        Self {
            ptr: ptr as *mut u8,
        }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` above, freed once.
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut core::ffi::c_void);
        }
    }
}

fn gb_per_s(bytes: usize, seconds: f64) -> f64 {
    bytes as f64 / seconds / 1e9
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// `(offset, bytes)` of each copy: `sizes` laid end to end.
fn layout(sizes: &[usize]) -> Vec<(usize, usize)> {
    let mut at = 0;
    sizes
        .iter()
        .map(|&s| {
            let o = at;
            at += s;
            (o, s)
        })
        .collect()
}

/// The median rate of `copy` over [`RUNS`] runs, each synchronized on `streams`.
fn measure(streams: &[std::sync::Arc<CudaStream>], total: usize, mut copy: impl FnMut()) -> f64 {
    let mut rates = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        for s in streams {
            s.synchronize().expect("idle");
        }
        let at = Instant::now();
        copy();
        for s in streams {
            s.synchronize().expect("copy");
        }
        rates.push(gb_per_s(total, at.elapsed().as_secs_f64()));
    }
    median(rates)
}

/// Pageable, pinned and pinned over two streams, for copies of `sizes`.
fn three_ways(
    host: &[u64],
    pinned: &Pinned,
    dev: &mut CudaSlice<u64>,
    s0: &std::sync::Arc<CudaStream>,
    s1: &std::sync::Arc<CudaStream>,
    sizes: &[usize],
) -> (f64, f64, f64) {
    let copies = layout(sizes);
    let total: usize = sizes.iter().sum();
    let pageable = measure(std::slice::from_ref(s0), total, || {
        for &(o, n) in &copies {
            let (e0, e1) = (o / 8, (o + n) / 8);
            let mut dst = dev.slice_mut(e0..e1);
            s0.memcpy_htod(&host[e0..e1], &mut dst)
                .expect("pageable htod");
        }
    });
    let (base, _record) = dev.device_ptr_mut(s0);
    let pin_on = |streams: &[&std::sync::Arc<CudaStream>]| {
        for (k, &(o, n)) in copies.iter().enumerate() {
            let s = streams[k % streams.len()];
            // SAFETY: both ranges lie inside allocations this function holds.
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    base + o as u64,
                    pinned.ptr.add(o) as *const core::ffi::c_void,
                    n,
                    s.cu_stream(),
                )
                .result()
                .expect("pinned htod");
            }
        }
    };
    let pinned_one = measure(std::slice::from_ref(s0), total, || pin_on(&[s0]));
    let both = [s0.clone(), s1.clone()];
    let pinned_two = measure(&both, total, || pin_on(&[s0, s1]));
    (pageable, pinned_one, pinned_two)
}

#[test]
#[ignore = "needs a card"]
fn pinned_against_pageable_at_column_sizes() {
    let be = backend().expect("a device");
    be.ctx.bind_to_thread().expect("bind");
    let s0 = be.next_stream();
    let s1 = be.next_stream();

    let epoch: Vec<usize> = EPOCH0
        .iter()
        .flat_map(|&(rows, cols)| std::iter::repeat_n(rows * 8, cols))
        .collect();
    let epoch_bytes: usize = epoch.iter().sum();
    let span = PER_SIZE_BYTES.max(epoch_bytes);
    let host: Vec<u64> = (0..span / 8)
        .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let pinned = Pinned::new(span);
    let mut dev = unsafe { math_cuda::device::alloc_or_trim::<u64>(&s0, span / 8) }.expect("alloc");

    println!(
        "\npinned vs pageable H2D, no host copy · median of {RUNS} · GB = 1e9 B\n\n\
         | columns | bytes | pageable GB/s | pinned GB/s | pinned x2 streams GB/s | pinned ÷ pageable |\n\
         |---|---|---|---|---|---|"
    );
    for &column in &[
        256usize << 10,
        1 << 20,
        4 << 20,
        16 << 20,
        32 << 20,
        64 << 20,
    ] {
        let sizes = vec![column; PER_SIZE_BYTES / column];
        let (pg, p1, p2) = three_ways(&host, &pinned, &mut dev, &s0, &s1, &sizes);
        println!(
            "| {} KiB × {} | {:.2} GB | {pg:.1} | {p1:.1} | {p2:.1} | {:.2}× |",
            column >> 10,
            sizes.len(),
            PER_SIZE_BYTES as f64 / 1e9,
            p1 / pg
        );
    }
    let (pg, p1, p2) = three_ways(&host, &pinned, &mut dev, &s0, &s1, &epoch);
    println!(
        "| epoch 0's {} columns | {:.2} GB | {pg:.1} | {p1:.1} | {p2:.1} | {:.2}× |",
        epoch.len(),
        epoch_bytes as f64 / 1e9,
        p1 / pg
    );
    let best = p1.max(p2);
    let verdict = if best >= 1.3 * pg {
        "WORTH IT (≥ 1.3×)"
    } else {
        "NOT WORTH IT (< 1.3×)"
    };
    println!(
        "\nMICROBENCH: epoch-0 mix pageable {pg:.1} GB/s · pinned {p1:.1} GB/s · pinned x2 {p2:.1} GB/s · \
         best ÷ pageable {:.2}× · stage 1b {verdict}",
        best / pg
    );
    assert!(pg > 0.0 && p1 > 0.0 && p2 > 0.0);
}
