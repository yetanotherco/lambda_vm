//! One LDE buffer: the fused row-major commits transpose their LDE to
//! column-major IN PLACE, so no second LDE-sized device allocation exists.
//!
//! The in-place transpose only moves whole runs of values, so its output must
//! equal, u64 for u64, the row-major host copy the very same call drains
//! before transposing. That is the pin here, for every entry point that
//! transposes: base, ext3 and the preprocessed split-tree pair. Roots against
//! the CPU commit are pinned by `merkle_root_parity`.
//!
//! `vram_arm` is the `#[ignore]`d measurement arm for the 10 Hz sampler at
//! the production shape (2^21 × 316 by default; env-overridable). Run it
//! with `LAMBDA_VM_MEMPOOL_RELEASE_MB=0` and the sampler attached:
//!
//! ```text
//! LAMBDA_VM_MEMPOOL_RELEASE_MB=0 cargo test -p math-cuda --release \
//!     --test one_lde_buffer vram_arm -- --ignored --nocapture
//! ```

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

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

fn random_row_major(rng: &mut ChaCha8Rng, n: usize, cols: usize) -> Vec<u64> {
    (0..n * cols).map(|_| rng.r#gen::<u64>()).collect()
}

/// D2H the handle's column-major buffer once its `ready` event has fired.
fn download(
    buf: &math_cuda::CudaSlice<u64>,
    ready: Option<&math_cuda::device::PooledEvent>,
) -> Vec<u64> {
    let be = math_cuda::device::backend().expect("cuda backend");
    let stream = be.next_stream();
    if let Some(ev) = ready {
        stream
            .wait(ev.event())
            .expect("wait on the handle's ready event");
    }
    stream.clone_dtoh(buf).expect("D2H of the column-major LDE")
}

/// `dev[c * rows + r] == host[r * cols + c]` for every element, raw u64s.
fn assert_column_major_of(dev: &[u64], host: &[u64], rows: usize, cols: usize, what: &str) {
    assert_eq!(dev.len(), rows * cols, "{what}: device length");
    assert_eq!(host.len(), rows * cols, "{what}: host length");
    for c in 0..cols {
        let col = &dev[c * rows..(c + 1) * rows];
        for (r, &v) in col.iter().enumerate() {
            let want = host[r * cols + c];
            assert_eq!(
                v, want,
                "{what}: row {r} col {c}: device {v:#x} != host {want:#x}"
            );
        }
    }
}

/// Shapes small enough to run in seconds yet wide enough to exercise the
/// multi-block run permutation (8 blocks from 2^3 rows up) with odd column
/// counts, a single column (identity), and both production blowups.
const SHAPES: &[(usize, usize, usize)] = &[
    // (log_n, blowup, cols)
    (1, 2, 2),
    (1, 4, 3),
    (2, 2, 7),
    (3, 2, 1),
    (3, 4, 5),
    (5, 2, 37),
    (6, 4, 3),
    (8, 2, 316),
    (9, 4, 37),
    (10, 2, 13),
    (12, 2, 316),
    (12, 4, 9),
];

#[test]
fn base_handle_is_the_in_place_transpose_of_the_host_lde() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let n = 1usize << log_n;
        let rows = n * blowup;
        let mut rng = ChaCha8Rng::seed_from_u64(0x1DE_0000 + i as u64);
        let row_major = random_row_major(&mut rng, n, cols);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let (handle, host_lde) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
            &row_major, None, n, cols, blowup, &weights, true,
        )
        .expect("fused base commit");
        assert_eq!(handle.m, cols);
        assert_eq!(handle.lde_size, rows);
        let dev = download(handle.buf.as_ref(), handle.ready.as_deref());
        assert_column_major_of(
            &dev,
            &host_lde,
            rows,
            cols,
            &format!("base log_n={log_n} blowup={blowup} cols={cols}"),
        );
        // The trace snapshot is the plain transpose of the input rows.
        let snap = handle
            .trace_dev
            .as_ref()
            .expect("base keep path retains the trace");
        let snap = download(snap.as_ref(), handle.ready.as_deref());
        assert_column_major_of(
            &snap,
            &row_major,
            n,
            cols,
            &format!("snapshot log_n={log_n} cols={cols}"),
        );
    }
}

#[test]
fn ext3_handle_is_the_in_place_transpose_of_the_host_lde() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        let n = 1usize << log_n;
        let rows = n * blowup;
        // `cols` ext3 columns = `3 * cols` base-field columns in row-major.
        let base_cols = cols * 3;
        let mut rng = ChaCha8Rng::seed_from_u64(0x31DE_0000 + i as u64);
        let row_major = random_row_major(&mut rng, n, base_cols);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let (handle, host_lde) = math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep(
            &row_major, n, cols, blowup, &weights, true,
        )
        .expect("fused ext3 commit");
        assert_eq!(handle.m, cols);
        assert_eq!(handle.lde_size, rows);
        let dev = download(handle.buf.as_ref(), handle.ready.as_deref());
        assert_column_major_of(
            &dev,
            &host_lde,
            rows,
            base_cols,
            &format!("ext3 log_n={log_n} blowup={blowup} cols={cols}"),
        );
    }
}

#[test]
fn split_tree_handle_is_the_in_place_transpose_of_the_host_lde() {
    for (i, &(log_n, blowup, cols)) in SHAPES.iter().enumerate() {
        if cols < 2 {
            continue; // the split needs a column on each side
        }
        let n = 1usize << log_n;
        let rows = n * blowup;
        let split_col = (cols / 2).max(1);
        let mut rng = ChaCha8Rng::seed_from_u64(0x51DE_0000 + i as u64);
        let row_major = random_row_major(&mut rng, n, cols);
        let weights = coset_weights_u64(n, COSET_OFFSET);
        let (pre_nodes, handle, host_lde) = math_cuda::lde::coset_lde_row_major_split_trees(
            &row_major, None, n, cols, blowup, &weights, split_col, true, true,
        )
        .expect("fused split-tree commit");
        assert!(pre_nodes.is_some(), "precomputed tree requested");
        let dev = download(handle.buf.as_ref(), handle.ready.as_deref());
        assert_column_major_of(
            &dev,
            &host_lde,
            rows,
            cols,
            &format!("split log_n={log_n} blowup={blowup} cols={cols}"),
        );
    }
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Deterministic, cheap fill for the gigabyte-scale arm (ChaCha would take
/// seconds per gigabyte).
fn splitmix_fill(seed: u64, out: &mut [u64]) {
    let mut x = seed;
    for v in out.iter_mut() {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *v = z ^ (z >> 31);
    }
}

/// Measurement arm for the 10 Hz VRAM sampler and the commit's wall time.
///
/// Env: `LAMBDA_VM_VRAM_ARM_LOG_N` (21), `_COLS` (316), `_BLOWUP` (2),
/// `_ITERS` (5), `_PREDEV` (unset: host
/// input, the brief's accounting; `1`: pre-uploaded device input as in
/// production, adds the trace to the resident set but times the commit
/// without the H2D). Prints the root (must not move between the old and the
/// new transpose), per-iteration wall time, and an in-process 1 kHz peak of
/// `used = total - free` device bytes as a cross-check on the sampler.
#[test]
#[ignore = "production-shape VRAM/time arm: minutes and gigabytes; run on the box with the sampler attached"]
fn vram_arm() {
    let log_n = env_or("LAMBDA_VM_VRAM_ARM_LOG_N", 21);
    let cols = env_or("LAMBDA_VM_VRAM_ARM_COLS", 316);
    let blowup = env_or("LAMBDA_VM_VRAM_ARM_BLOWUP", 2);
    let iters = env_or("LAMBDA_VM_VRAM_ARM_ITERS", 5).max(1);
    let predev = std::env::var_os("LAMBDA_VM_VRAM_ARM_PREDEV").is_some();
    let n = 1usize << log_n;
    let rows = n * blowup;
    let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
    let lde_bytes = rows * cols * 8;
    let trace_bytes = n * cols * 8;
    let tree_bytes = (rows - 1) * 32;
    println!(
        "vram_arm: 2^{log_n} rows x {cols} cols @ blowup {blowup}, input {}",
        if predev { "device (predev)" } else { "host" }
    );
    println!(
        "vram_arm: model  LDE {:.2} GiB  snapshot {:.2} GiB  tree {:.3} GiB  => one-buffer floor {:.2} GiB, two-buffer {:.2} GiB{}",
        gib(lde_bytes),
        gib(trace_bytes),
        gib(tree_bytes),
        gib(lde_bytes + trace_bytes + tree_bytes),
        gib(2 * lde_bytes + trace_bytes + tree_bytes),
        if predev {
            format!(" (+ {:.2} GiB resident predev)", gib(trace_bytes))
        } else {
            String::new()
        }
    );

    let mut row_major = vec![0u64; n * cols];
    splitmix_fill(0x1DE_2026_0907, &mut row_major);
    let weights = coset_weights_u64(n, COSET_OFFSET);

    let be = math_cuda::device::backend().expect("cuda backend");
    let predev_buf = predev.then(|| {
        let stream = be.next_stream();
        let d = stream.clone_htod(&row_major).expect("predev upload");
        stream.synchronize().expect("predev upload sync");
        d
    });

    // In-process peak sampler: the primary context is shared, so this costs
    // no device memory of its own.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sampler = {
        let (stop, peak) = (stop.clone(), peak.clone());
        std::thread::spawn(move || {
            let ctx = cudarc::driver::CudaContext::new(0).expect("primary context");
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok((free, total)) = ctx.mem_get_info() {
                    peak.fetch_max(total - free, std::sync::atomic::Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
    };
    let (baseline_free, total) = be.ctx.mem_get_info().expect("mem_get_info");
    println!(
        "vram_arm: device used before the loop {:.2} GiB of {:.2} GiB",
        gib(total - baseline_free),
        gib(total)
    );

    let mut first_root = None;
    for it in 0..iters {
        let t0 = std::time::Instant::now();
        let (handle, host) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
            &row_major,
            predev_buf.as_ref(),
            n,
            cols,
            blowup,
            &weights,
            false,
        )
        .expect("fused commit");
        // The handle's `ready` fires after the transpose; wait so the timing
        // covers the whole commit and the peak is held while we look.
        let stream = be.next_stream();
        handle.wait_ready_on(&stream).expect("wait ready");
        stream.synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        assert!(host.is_empty(), "device-only commit returns no host LDE");
        let root = handle.tree.as_ref().expect("resident tree").root;
        let hex: String = root.iter().map(|b| format!("{b:02x}")).collect();
        match &first_root {
            None => first_root = Some(root),
            Some(r) => assert_eq!(*r, root, "root moved between iterations"),
        }
        println!("vram_arm: iter {it}: {ms:.1} ms  root {hex}");
        drop(handle);
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().expect("sampler thread");
    println!(
        "vram_arm: in-process peak used {:.2} GiB (1 kHz, total - free; includes the context)",
        gib(peak.load(std::sync::atomic::Ordering::Relaxed))
    );
}
