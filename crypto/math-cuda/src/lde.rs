//! Full coset LDE on device. Mirrors `Polynomial::coset_lde_full_expand` in
//! `crypto/math/src/fft/polynomial.rs` algebraically:
//!
//! Input  : N evaluations (natural order) of a poly on the standard subgroup,
//!          plus coset weights (size N). The weights include the `1/N` iFFT
//!          normalisation, matching the `LdeTwiddles::coset_weights` format at
//!          `crypto/stark/src/prover.rs:248` — i.e. `weights[i] = g^i / N`.
//! Output : N*blowup_factor evaluations (natural order) on the coset.
//!
//! On-device steps, picks a stream from the shared pool so rayon-parallel
//! callers overlap on the GPU. Twiddles are cached in the backend.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::ntt::run_ntt_body;

pub fn coset_lde_base(
    evals: &[u64],
    blowup_factor: usize,
    weights: &[u64],
) -> Result<Vec<u64>> {
    let n = evals.len();
    assert!(n.is_power_of_two(), "evals length must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match evals");
    assert!(blowup_factor.is_power_of_two(), "blowup must be power of two");
    if n == 0 {
        return Ok(Vec::new());
    }
    let lde_size = n * blowup_factor;
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend();
    let stream = be.next_stream();

    // Device buffer of lde_size, zero-padded tail, first N filled by copy.
    let mut buf = stream.alloc_zeros::<u64>(lde_size)?;
    {
        let mut head = buf.slice_mut(0..n);
        stream.memcpy_htod(evals, &mut head)?;
    }

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;

    // === 1. iNTT on first N: bit_reverse + 8-level-fused DIT body ===
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute)
            .arg(&mut buf)
            .arg(&n_u64)
            .arg(&log_n)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    // Note: `run_ntt_body` expects a standalone CudaSlice; we pass `buf` and
    // the kernel walks the first `n_u64` elements via its own indexing.
    run_ntt_body(stream.as_ref(), &mut buf, inv_tw.as_ref(), n_u64, log_n)?;
    // Note: the CPU iFFT does not include 1/N — it's folded into `weights`. The
    // next pointwise multiply applies both the coset shift and the 1/N factor.

    // === 2. Pointwise multiply first N by coset weights (includes 1/N) ===
    unsafe {
        stream
            .launch_builder(&be.pointwise_mul)
            .arg(&mut buf)
            .arg(&weights_dev)
            .arg(&n_u64)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    // === 3. Forward NTT on full buffer ===
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute)
            .arg(&mut buf)
            .arg(&lde_u64)
            .arg(&log_lde)
            .launch(LaunchConfig::for_num_elems(lde_size as u32))?;
    }
    run_ntt_body(stream.as_ref(), &mut buf, fwd_tw.as_ref(), lde_u64, log_lde)?;

    let out = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(out)
}

/// Batched coset LDE: processes `m` columns (all the same domain) in a single
/// pipeline on one stream. One H2D per column, then per-level batched kernels
/// that launch with `grid.y = m` so a single launch does the butterflies for
/// every column at that level.
///
/// Returns one `Vec<u64>` per input column, each of length `n * blowup_factor`.
pub fn coset_lde_batch_base(
    columns: &[&[u64]],
    blowup_factor: usize,
    weights: &[u64],
) -> Result<Vec<Vec<u64>>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let m = columns.len();
    let n = columns[0].len();
    assert!(n.is_power_of_two(), "column length must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match column length");
    assert!(blowup_factor.is_power_of_two(), "blowup must be power of two");
    for c in columns.iter() {
        assert_eq!(c.len(), n, "all columns must be the same size");
    }

    if n == 0 {
        return Ok(vec![Vec::new(); m]);
    }
    let lde_size = n * blowup_factor;
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend();
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    let debug_phases = std::env::var("MATH_CUDA_PHASE_TIMING").is_ok();
    let t_start = if debug_phases { Some(std::time::Instant::now()) } else { None };
    let phase = |label: &str, prev: &mut Option<std::time::Instant>| {
        if let Some(p) = prev.as_ref() {
            let now = std::time::Instant::now();
            eprintln!("  [{:>6.2} ms] {}", (now - *p).as_secs_f64() * 1e3, label);
            *prev = Some(now);
        }
    };
    let mut last = t_start;

    // Pinned staging. Lock and grow to max(m*n for upload, m*lde_size for
    // download). Holding the guard across the whole call serialises concurrent
    // batched calls that happened to hash to the same stream slot, but that's
    // exactly what we want — one stream can only do one sequence at a time.
    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(m * lde_size, &be.ctx)?;
    // SAFETY: staging is locked, the slice alias ends before we unlock.
    let pinned = unsafe { staging.as_mut_slice(m * lde_size) };
    if debug_phases { phase("staging lock + grow", &mut last); }

    // Pack columns into first m*n slots of the pinned buffer, then one big H2D.
    for (c, col) in columns.iter().enumerate() {
        pinned[c * n..c * n + n].copy_from_slice(col);
    }
    if debug_phases { phase("host pack (pinned)", &mut last); }

    // Column layout: `buf[c * lde_size + r]`. Zeroed so the [n, lde_size)
    // tail of each column is already the zero-pad the CPU path does.
    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    if debug_phases { stream.synchronize()?; phase("alloc_zeros", &mut last); }
    // One memcpy per column from the pinned buffer into the strided slots.
    // The pinned source hits PCIe line-rate.
    for c in 0..m {
        let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
        stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
    }
    if debug_phases { stream.synchronize()?; phase("H2D cols (pinned)", &mut last); }

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;
    if debug_phases { stream.synchronize()?; phase("twiddles + weights", &mut last); }

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let m_u32 = m as u32;

    // === 1. Bit-reverse first N of every column ===
    {
        let grid_x = (n as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute_batched)
                .arg(&mut buf)
                .arg(&n_u64)
                .arg(&log_n)
                .arg(&col_stride_u64)
                .launch(cfg)?;
        }
    }

    if debug_phases { stream.synchronize()?; phase("bit_reverse N", &mut last); }
    // === 2. iNTT body over all columns ===
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;
    if debug_phases { stream.synchronize()?; phase("iNTT body", &mut last); }

    // === 3. Pointwise multiply by coset weights (includes 1/N) ===
    {
        let grid_x = (n as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.pointwise_mul_batched)
                .arg(&mut buf)
                .arg(&weights_dev)
                .arg(&n_u64)
                .arg(&col_stride_u64)
                .launch(cfg)?;
        }
    }

    // === 4. Bit-reverse full LDE of every column ===
    {
        let grid_x = (lde_size as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute_batched)
                .arg(&mut buf)
                .arg(&lde_u64)
                .arg(&log_lde)
                .arg(&col_stride_u64)
                .launch(cfg)?;
        }
    }

    if debug_phases { stream.synchronize()?; phase("pointwise + bit_reverse LDE", &mut last); }
    // === 5. Forward NTT on full LDE of every column ===
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;
    if debug_phases { stream.synchronize()?; phase("forward NTT body", &mut last); }

    // Single big D2H into the reusable pinned staging buffer — pinned, one
    // call to the driver, saturates PCIe.
    stream.memcpy_dtoh(&buf, &mut pinned[..m * lde_size])?;
    stream.synchronize()?;
    if debug_phases { phase("D2H (one shot into pinned)", &mut last); }

    // Split pinned → per-column Vec<u64>s. The first write to each virgin
    // Vec page-faults, which can dominate total time (~75 ms for 128 MB).
    // Parallelise so the fault cost spreads across CPU cores.
    use rayon::prelude::*;
    let pinned_ptr = pinned.as_ptr() as usize; // Send a usize to dodge aliasing rules.
    let out: Vec<Vec<u64>> = (0..m)
        .into_par_iter()
        .map(|c| {
            let mut v = Vec::<u64>::with_capacity(lde_size);
            // SAFETY: we overwrite the entire range immediately below.
            unsafe { v.set_len(lde_size) };
            // SAFETY: pinned buffer is held locked by the caller (staging
            // guard); the slice doesn't escape and can't alias another
            // column's write since `v` is thread-local.
            let src = unsafe {
                std::slice::from_raw_parts(
                    (pinned_ptr as *const u64).add(c * lde_size),
                    lde_size,
                )
            };
            v.copy_from_slice(src);
            v
        })
        .collect();
    if debug_phases { phase("copy out (rayon pinned → Vecs)", &mut last); }
    drop(staging);
    Ok(out)
}

/// Like `coset_lde_batch_base` but writes directly into caller-provided
/// output slices instead of allocating fresh `Vec<u64>`s. Each output slice
/// must already have length `n * blowup_factor`. Saves ~50–100 ms of pageable
/// allocator work + page faults at prover scale because the caller's Vecs
/// have been sized once and are reused across calls.
pub fn coset_lde_batch_base_into(
    columns: &[&[u64]],
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
) -> Result<()> {
    if columns.is_empty() {
        return Ok(());
    }
    let m = columns.len();
    assert_eq!(outputs.len(), m, "outputs must match columns count");
    let n = columns[0].len();
    assert!(n.is_power_of_two(), "column length must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match column length");
    assert!(blowup_factor.is_power_of_two(), "blowup must be power of two");
    for c in columns.iter() {
        assert_eq!(c.len(), n, "all columns must be the same size");
    }
    let lde_size = n * blowup_factor;
    for o in outputs.iter() {
        assert_eq!(o.len(), lde_size, "each output must be lde_size");
    }
    if n == 0 {
        return Ok(());
    }
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend();
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(m * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(m * lde_size) };

    for (c, col) in columns.iter().enumerate() {
        pinned[c * n..c * n + n].copy_from_slice(col);
    }

    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    for c in 0..m {
        let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
        stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
    }

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let m_u32 = m as u32;

    // iNTT bit-reverse + body, pointwise mul, forward bit-reverse + body.
    {
        let grid_x = (n as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute_batched)
                .arg(&mut buf)
                .arg(&n_u64)
                .arg(&log_n)
                .arg(&col_stride_u64)
                .launch(cfg)?;
        }
    }
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;
    {
        let grid_x = (n as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.pointwise_mul_batched)
                .arg(&mut buf)
                .arg(&weights_dev)
                .arg(&n_u64)
                .arg(&col_stride_u64)
                .launch(cfg)?;
        }
    }
    {
        let grid_x = (lde_size as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute_batched)
                .arg(&mut buf)
                .arg(&lde_u64)
                .arg(&log_lde)
                .arg(&col_stride_u64)
                .launch(cfg)?;
        }
    }
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;

    stream.memcpy_dtoh(&buf, &mut pinned[..m * lde_size])?;
    stream.synchronize()?;

    // Parallel copy pinned → caller outputs. Caller's Vecs should already be
    // faulted/resized so no page-fault cost here.
    use rayon::prelude::*;
    let pinned_ptr = pinned.as_ptr() as usize;
    outputs
        .par_iter_mut()
        .enumerate()
        .for_each(|(c, dst)| {
            let src = unsafe {
                std::slice::from_raw_parts(
                    (pinned_ptr as *const u64).add(c * lde_size),
                    lde_size,
                )
            };
            dst.copy_from_slice(src);
        });
    drop(staging);
    Ok(())
}

/// Run the DIT butterfly body of a bit-reversed-input NTT over `m` batched
/// columns in one device buffer. Same fusion strategy as `run_ntt_body`:
/// first 8 levels shmem-fused (coalesced), subsequent levels one kernel each.
fn run_batched_ntt_body(
    stream: &cudarc::driver::CudaStream,
    x_dev: &mut cudarc::driver::CudaSlice<u64>,
    tw_dev: &cudarc::driver::CudaSlice<u64>,
    n: u64,
    log_n: u64,
    col_stride: u64,
    m: u32,
) -> Result<()> {
    let be = backend();
    let fused = core::cmp::min(log_n, 8);
    if fused >= 8 {
        let grid_x = (n / 256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let base_step = 0u64;
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_8_levels_batched)
                .arg(&mut *x_dev)
                .arg(tw_dev)
                .arg(&n)
                .arg(&log_n)
                .arg(&base_step)
                .arg(&col_stride)
                .launch(cfg)?;
        }
    } else {
        let grid_x = ((n / 2) as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, m, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        for level in 0..fused {
            unsafe {
                stream
                    .launch_builder(&be.ntt_dit_level_batched)
                    .arg(&mut *x_dev)
                    .arg(tw_dev)
                    .arg(&n)
                    .arg(&log_n)
                    .arg(&level)
                    .arg(&col_stride)
                    .launch(cfg)?;
            }
        }
    }

    let grid_x = ((n / 2) as u32).div_ceil(256).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, m, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    for level in fused..log_n {
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_level_batched)
                .arg(&mut *x_dev)
                .arg(tw_dev)
                .arg(&n)
                .arg(&log_n)
                .arg(&level)
                .arg(&col_stride)
                .launch(cfg)?;
        }
    }
    Ok(())
}

