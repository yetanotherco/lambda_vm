//! Multi-field-key multiplicity counting on device. Takes parallel
//! `(keys_hi, keys_lo)` u64 arrays packing each operation's multi-field
//! key into a u128, sorts them via 128-pass bit-by-bit radix sort, then
//! does a segmented reduce to produce a dense `(unique_keys, counts)`
//! pair.
//!
//! Kernels live in `kernels/multiplicity_sort.cu`. The radix sort loop
//! and segmented reduce are stream-internal (no host bounce except for one
//! u64 D2H at the end to learn `num_unique` so the caller can size its
//! output buffers).
//!
//! Performance estimate (n = 4M keys on RTX 5090, HBM ≈ 1.5 TB/s):
//!   - 128 sort passes × (predicate + scan + scatter): ~25 ms
//!   - Segmented reduce: ~5 ms
//!   - Total: ~30 ms

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{Backend, backend};

const BLOCK_SIZE: u32 = 256;

fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check_grid_bound(n: usize) -> Result<()> {
    if n > u32::MAX as usize / BLOCK_SIZE as usize {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ));
    }
    Ok(())
}

/// Result of [`multiplicity_count_multifield_dev`]: dense parallel arrays
/// of `num_unique` entries each. The order of unique keys is sorted
/// ascending by `(hi, lo)` (stable, since the radix sort is bit-by-bit
/// LSB-first).
pub struct MultiFieldCountResult {
    pub unique_hi: CudaSlice<u64>,
    pub unique_lo: CudaSlice<u64>,
    pub counts: CudaSlice<u64>,
    pub num_unique: usize,
}

// ===========================================================================
// Forward inclusive scan over u64 add (recursive driver, mirrors PR-5's
// ext3-mul scan structure but on plain u64).
// ===========================================================================

/// Inclusive prefix sum of `input` into `out`. Scratch allocations are
/// internal. Top-level call: `input` and `out` must NOT alias.
fn inclusive_scan_into_fwd_u64(
    stream: &Arc<CudaStream>,
    be: &Backend,
    input: &CudaSlice<u64>,
    out: &mut CudaSlice<u64>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let k = (n as u32).div_ceil(BLOCK_SIZE);
    let mut block_totals = unsafe { stream.alloc::<u64>(k as usize) }?;
    let n_u = n as u64;
    let phase_cfg = LaunchConfig {
        grid_dim: (k, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&be.block_inclusive_scan_fwd_u64)
            .arg(input)
            .arg(&n_u)
            .arg(&mut *out)
            .arg(&mut block_totals)
            .launch(phase_cfg)?;
    }

    if k > 1 {
        // Recursively scan block_totals in place via scratch buffer.
        inclusive_scan_inplace_fwd_u64(stream, be, &mut block_totals, k as usize)?;
        unsafe {
            stream
                .launch_builder(&be.apply_block_offsets_fwd_u64)
                .arg(&mut *out)
                .arg(&n_u)
                .arg(&block_totals)
                .launch(phase_cfg)?;
        }
    }
    Ok(())
}

/// In-place inclusive scan. Used by the recursion when scanning block
/// totals (same buffer for read and write). Same scratch+memcpy pattern
/// as inverse.rs's `scan_inplace_fwd`: cudarc's borrow checker won't let
/// us pass the same buffer as both read and write args.
fn inclusive_scan_inplace_fwd_u64(
    stream: &Arc<CudaStream>,
    be: &Backend,
    buf: &mut CudaSlice<u64>,
    n: usize,
) -> Result<()> {
    if n <= 1 {
        return Ok(());
    }
    let k = (n as u32).div_ceil(BLOCK_SIZE);
    let mut block_totals = unsafe { stream.alloc::<u64>(k as usize) }?;
    let n_u = n as u64;
    let phase_cfg = LaunchConfig {
        grid_dim: (k, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut scratch = unsafe { stream.alloc::<u64>(n) }?;
    unsafe {
        stream
            .launch_builder(&be.block_inclusive_scan_fwd_u64)
            .arg(&*buf)
            .arg(&n_u)
            .arg(&mut scratch)
            .arg(&mut block_totals)
            .launch(phase_cfg)?;
    }
    stream.memcpy_dtod(&scratch, buf)?;

    if k > 1 {
        inclusive_scan_inplace_fwd_u64(stream, be, &mut block_totals, k as usize)?;
        unsafe {
            stream
                .launch_builder(&be.apply_block_offsets_fwd_u64)
                .arg(&mut *buf)
                .arg(&n_u)
                .arg(&block_totals)
                .launch(phase_cfg)?;
        }
    }
    Ok(())
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Sort the `(keys_hi, keys_lo)` u128 keys and return a dense list of
/// unique keys + their multiplicities. Stable order; sorted ascending
/// by `(hi, lo)`.
///
/// Returns `Err` on grid-bound overflow (`n > u32::MAX / BLOCK_SIZE`) or
/// any cudarc error. Empty input → zero-length result.
pub fn multiplicity_count_multifield_dev(
    keys_hi: &CudaSlice<u64>,
    keys_lo: &CudaSlice<u64>,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<MultiFieldCountResult> {
    assert!(keys_hi.len() >= n);
    assert!(keys_lo.len() >= n);
    check_grid_bound(n)?;

    if n == 0 {
        return Ok(MultiFieldCountResult {
            unique_hi: unsafe { stream.alloc::<u64>(0) }?,
            unique_lo: unsafe { stream.alloc::<u64>(0) }?,
            counts: unsafe { stream.alloc::<u64>(0) }?,
            num_unique: 0,
        });
    }

    let be = backend()?;
    let n_u = n as u64;

    // Ping-pong key buffers. The "in" pair starts as a copy of the
    // caller-supplied keys (we don't want to mutate them).
    // SAFETY: every byte of *_a/*_b is overwritten on first use.
    let mut hi_a = unsafe { stream.alloc::<u64>(n) }?;
    let mut lo_a = unsafe { stream.alloc::<u64>(n) }?;
    let mut hi_b = unsafe { stream.alloc::<u64>(n) }?;
    let mut lo_b = unsafe { stream.alloc::<u64>(n) }?;
    stream.memcpy_dtod(keys_hi, &mut hi_a)?;
    stream.memcpy_dtod(keys_lo, &mut lo_a)?;
    let mut a_is_input = true;

    let mut pred = unsafe { stream.alloc::<u64>(n) }?;
    let mut scan_buf = unsafe { stream.alloc::<u64>(n) }?;

    // Stage A: 128 bit-passes (LSB through lo, then through hi). Bit-by-bit
    // radix sort is stable, which we rely on for the final ordering.
    for bit in 0..128u32 {
        let (hi_in, lo_in, hi_out, lo_out) = if a_is_input {
            (&hi_a, &lo_a, &mut hi_b, &mut lo_b)
        } else {
            (&hi_b, &lo_b, &mut hi_a, &mut lo_a)
        };

        // 1. predicate per element
        unsafe {
            stream
                .launch_builder(&be.extract_bit_predicate)
                .arg(hi_in)
                .arg(lo_in)
                .arg(&n_u)
                .arg(&bit)
                .arg(&mut pred)
                .launch(launch_cfg(n))?;
        }
        // 2. inclusive scan of pred into scan_buf
        inclusive_scan_into_fwd_u64(stream, be, &pred, &mut scan_buf, n)?;
        // 3. scatter (reads total_ones = scan_buf[n-1] on device)
        unsafe {
            stream
                .launch_builder(&be.scatter_by_bit)
                .arg(hi_in)
                .arg(lo_in)
                .arg(&n_u)
                .arg(&pred)
                .arg(&scan_buf)
                .arg(hi_out)
                .arg(lo_out)
                .launch(launch_cfg(n))?;
        }
        a_is_input = !a_is_input;
    }

    // After 128 passes the sorted keys are in whichever buffer last received
    // the scatter output. If `a_is_input` was flipped 128 times (even count)
    // we end up back where we started — so the SORTED data is in *_a.
    // Wait: we flip after each scatter. 128 flips means final state is the
    // SAME as initial (a_is_input == true). Each pass: read from "in",
    // write to "out", then flip. After 128 passes, last write was to:
    //   - pass 0: a→b, then a_is_input=false
    //   - pass 1: b→a, then a_is_input=true
    //   - pass 127 (odd): b→a, then a_is_input=false
    //   - pass 127 (since starts at 0): bit=127, that's the 128th pass
    //     (i = 0..=127, count = 128). Final flip: a_is_input was true at
    //     start of pass 0, flipped 128 times → true at end. So the last
    //     scatter wrote to `b` (when a_is_input was true going in).
    // The sorted output is in *_b.
    debug_assert!(a_is_input, "ping-pong invariant broken");
    let sorted_hi = &hi_b;
    let sorted_lo = &lo_b;

    // Stage B: segmented reduce.
    let mut is_first = unsafe { stream.alloc::<u64>(n) }?;
    unsafe {
        stream
            .launch_builder(&be.mark_boundaries)
            .arg(sorted_hi)
            .arg(sorted_lo)
            .arg(&n_u)
            .arg(&mut is_first)
            .launch(launch_cfg(n))?;
    }

    let mut first_scan = unsafe { stream.alloc::<u64>(n) }?;
    inclusive_scan_into_fwd_u64(stream, be, &is_first, &mut first_scan, n)?;

    // D2H num_unique = first_scan[n-1].
    let last: Vec<u64> = stream.clone_dtoh(&first_scan.slice((n - 1)..n))?;
    stream.synchronize()?;
    let num_unique = last[0] as usize;

    let mut unique_hi = unsafe { stream.alloc::<u64>(num_unique) }?;
    let mut unique_lo = unsafe { stream.alloc::<u64>(num_unique) }?;
    let mut counts = stream.alloc_zeros::<u64>(num_unique)?;

    unsafe {
        stream
            .launch_builder(&be.compact_unique_and_counts)
            .arg(sorted_hi)
            .arg(sorted_lo)
            .arg(&n_u)
            .arg(&is_first)
            .arg(&first_scan)
            .arg(&mut unique_hi)
            .arg(&mut unique_lo)
            .arg(&mut counts)
            .launch(launch_cfg(n))?;
    }

    Ok(MultiFieldCountResult {
        unique_hi,
        unique_lo,
        counts,
        num_unique,
    })
}
