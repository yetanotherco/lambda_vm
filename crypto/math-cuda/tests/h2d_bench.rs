//! What it costs to put a commitment's coefficients on the card.
//!
//! The commit is the biggest thing this crate crosses the bus with — 42.56 GiB
//! over a proof of the real block, in 1329 uploads — so what it runs at decides
//! a couple of seconds. This is the measurement behind the log's §(bb).
//!
//! Three shapes, all measured on the box:
//!
//! | | GiB/s |
//! |---|---:|
//! | pageable, one stream (what the commit does) | 18.6 |
//! | page-locked source | 23.7 |
//! | chunked through a pinned buffer we own, two in flight | 13.9 |
//!
//! In the proof the pageable path drops to **12.9 GiB/s** — two commits upload
//! at once and share the driver's staging buffer — and the page-locked copy
//! rises to **39**. So the copy itself could be 1.1 s instead of 3.3.
//!
//! **What stops it**: page-locking a range on the fly costs 0.95 s to register
//! and 1.10 to unregister over the same 42.56 GiB, which is the whole gain.
//! Raising the size threshold does not help — 99.6% of the bytes are already in
//! commits of 64 MiB or more. The win needs the stacked polynomial to be
//! **allocated** pinned rather than page-locked after the fact.
//!
//! Server-only; needs a real device. `cargo test --release -p math-cuda
//! --test h2d_bench -- --ignored --nocapture`.

use std::time::Instant;

const LOG_ELEMS: usize = 25;
const RUNS: usize = 5;

fn gib_per_s(bytes: usize, seconds: f64) -> f64 {
    bytes as f64 / seconds / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore]
fn upload_costs() {
    let n = 1usize << LOG_ELEMS;
    let bytes = n * 8;
    let host: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15))
        .collect();

    let be = math_cuda::device::backend().expect("a device");
    let stream = be.next_stream();

    println!(
        "\n{} MiB per upload, {RUNS} runs each\n",
        bytes / (1024 * 1024)
    );

    let mut plain = 0.0;
    for _ in 0..RUNS {
        let mut dev =
            unsafe { math_cuda::device::alloc_or_trim::<u64>(&stream, n) }.expect("alloc");
        stream.synchronize().expect("idle");
        let at = Instant::now();
        stream.memcpy_htod(&host, &mut dev).expect("htod");
        stream.synchronize().expect("copy");
        plain += at.elapsed().as_secs_f64();
    }
    println!(
        "pageable      {:>7.3} s   {:>6.2} GiB/s",
        plain / RUNS as f64,
        gib_per_s(bytes * RUNS, plain)
    );

    // Registering walks the range's page table, so it is timed apart: that is
    // what decides whether page-locking on the fly is worth anything, and in
    // the proof it is not.
    let mut registering = 0.0;
    let mut copying = 0.0;
    let mut unregistering = 0.0;
    let at_ptr = host.as_ptr() as *mut core::ffi::c_void;
    for _ in 0..RUNS {
        let mut dev =
            unsafe { math_cuda::device::alloc_or_trim::<u64>(&stream, n) }.expect("alloc");
        stream.synchronize().expect("idle");

        let at = Instant::now();
        // SAFETY: `host` outlives the registration, which is undone below.
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(at_ptr, bytes, 0)
                .result()
                .expect("register");
        }
        registering += at.elapsed().as_secs_f64();

        let at = Instant::now();
        stream.memcpy_htod(&host, &mut dev).expect("htod");
        stream.synchronize().expect("copy");
        copying += at.elapsed().as_secs_f64();

        let at = Instant::now();
        // SAFETY: registered just above, undone exactly once.
        unsafe {
            cudarc::driver::sys::cuMemHostUnregister(at_ptr)
                .result()
                .expect("unregister");
        }
        unregistering += at.elapsed().as_secs_f64();
    }
    println!(
        "page-locked   {:>7.3} s   {:>6.2} GiB/s   (register {:.3} s, unregister {:.3} s)",
        (registering + copying + unregistering) / RUNS as f64,
        gib_per_s(bytes * RUNS, registering + copying + unregistering),
        registering / RUNS as f64,
        unregistering / RUNS as f64,
    );
    println!(
        "  the copy alone         {:>6.2} GiB/s — what an already-pinned source would give",
        gib_per_s(bytes * RUNS, copying)
    );
}
