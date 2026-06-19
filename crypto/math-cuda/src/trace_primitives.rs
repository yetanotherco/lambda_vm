//! Trivial trace-builder primitives — bulk u64 shuffles reused by every
//! per-table GPU port (page, decode, future ones).
//!
//! Kernels live in `kernels/trace_primitives.cu`. Each host wrapper exposes
//! a device-input → device-output form (caller threads its own stream so
//! producer + consumer kernels serialize naturally) and, for the host-input
//! variants used by tests + bootstrap call sites, a convenience wrapper
//! that H2Ds once + returns a Vec.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Runtime guard against `(n as u32).div_ceil(BLOCK_SIZE)` truncation on
/// pathologically large inputs. Returns `Err` so the caller's CPU fallback
/// fires (PR-5 set this precedent for `batch_inverse_ext3_dev`).
fn check_grid_bound(n: usize) -> Result<()> {
    if n > u32::MAX as usize / BLOCK_SIZE as usize {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ));
    }
    Ok(())
}

// ===========================================================================
// 1. pad_to_pow2_u64
// ===========================================================================

/// `dst[i] = src[i] if i < src_len else sentinel`, for i in 0..dst_len.
/// Allocates the output buffer on the caller's stream.
pub fn pad_to_pow2_u64_dev(
    src: &CudaSlice<u64>,
    src_len: usize,
    sentinel: u64,
    dst_len: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    assert!(dst_len >= src_len);
    assert!(src.len() >= src_len);
    check_grid_bound(dst_len)?;
    let be = backend()?;
    // SAFETY: the kernel writes every output slot.
    let mut dst = unsafe { stream.alloc::<u64>(dst_len) }?;
    if dst_len == 0 {
        return Ok(dst);
    }
    let src_len_u = src_len as u64;
    let dst_len_u = dst_len as u64;
    unsafe {
        stream
            .launch_builder(&be.pad_to_pow2_u64)
            .arg(src)
            .arg(&src_len_u)
            .arg(&sentinel)
            .arg(&mut dst)
            .arg(&dst_len_u)
            .launch(launch_cfg(dst_len))?;
    }
    Ok(dst)
}

// ===========================================================================
// 2. decompose_u64_to_bytes
// ===========================================================================

/// For each `src[i]`, writes 8 LSB-first bytes to `dst[i*8 .. i*8+8]`.
/// Returned slice has length `8 * n`.
pub fn decompose_u64_to_bytes_dev(
    src: &CudaSlice<u64>,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    assert!(src.len() >= n);
    check_grid_bound(n)?;
    let be = backend()?;
    // SAFETY: the kernel writes every output slot.
    let mut dst = unsafe { stream.alloc::<u64>(n * 8) }?;
    if n == 0 {
        return Ok(dst);
    }
    let n_u = n as u64;
    unsafe {
        stream
            .launch_builder(&be.decompose_u64_to_bytes)
            .arg(src)
            .arg(&n_u)
            .arg(&mut dst)
            .launch(launch_cfg(n))?;
    }
    Ok(dst)
}

// ===========================================================================
// 3. decompose_u64_to_halfwords
// ===========================================================================

/// For each `src[i]`, writes 4 LSB-first 16-bit halfwords to
/// `dst[i*4 .. i*4+4]`. Returned slice has length `4 * n`.
pub fn decompose_u64_to_halfwords_dev(
    src: &CudaSlice<u64>,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    assert!(src.len() >= n);
    check_grid_bound(n)?;
    let be = backend()?;
    let mut dst = unsafe { stream.alloc::<u64>(n * 4) }?;
    if n == 0 {
        return Ok(dst);
    }
    let n_u = n as u64;
    unsafe {
        stream
            .launch_builder(&be.decompose_u64_to_halfwords)
            .arg(src)
            .arg(&n_u)
            .arg(&mut dst)
            .launch(launch_cfg(n))?;
    }
    Ok(dst)
}

// ===========================================================================
// 4. fill_sequential_u64
// ===========================================================================

/// `dst[i] = start + i * stride` (plain u64 arithmetic, no field reduction).
pub fn fill_sequential_u64_dev(
    start: u64,
    stride: u64,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    check_grid_bound(n)?;
    let be = backend()?;
    let mut dst = unsafe { stream.alloc::<u64>(n) }?;
    if n == 0 {
        return Ok(dst);
    }
    let n_u = n as u64;
    unsafe {
        stream
            .launch_builder(&be.fill_sequential_u64)
            .arg(&start)
            .arg(&stride)
            .arg(&n_u)
            .arg(&mut dst)
            .launch(launch_cfg(n))?;
    }
    Ok(dst)
}

// ===========================================================================
// 5. range_check_column_u64
// ===========================================================================

/// `dst[i] = i` for i in 0..n.
pub fn range_check_column_u64_dev(
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    check_grid_bound(n)?;
    let be = backend()?;
    let mut dst = unsafe { stream.alloc::<u64>(n) }?;
    if n == 0 {
        return Ok(dst);
    }
    let n_u = n as u64;
    unsafe {
        stream
            .launch_builder(&be.range_check_column_u64)
            .arg(&n_u)
            .arg(&mut dst)
            .launch(launch_cfg(n))?;
    }
    Ok(dst)
}

// ===========================================================================
// 6. extract_bits_u64
// ===========================================================================

/// `dst[i] = (src[i] >> shift) & ((1 << width) - 1)`. `width == 64` is a
/// no-mask passthrough of `src >> shift`.
pub fn extract_bits_u64_dev(
    src: &CudaSlice<u64>,
    n: usize,
    shift: u32,
    width: u32,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    assert!(src.len() >= n);
    assert!(shift < 64, "shift must be < 64");
    assert!((1..=64).contains(&width), "width must be in 1..=64");
    check_grid_bound(n)?;
    let be = backend()?;
    let mut dst = unsafe { stream.alloc::<u64>(n) }?;
    if n == 0 {
        return Ok(dst);
    }
    let n_u = n as u64;
    unsafe {
        stream
            .launch_builder(&be.extract_bits_u64)
            .arg(src)
            .arg(&n_u)
            .arg(&shift)
            .arg(&width)
            .arg(&mut dst)
            .launch(launch_cfg(n))?;
    }
    Ok(dst)
}

// ===========================================================================
// 7. multiplicity_count_by_index
// ===========================================================================

/// `counts[keys[i]] += 1` for each i in 0..n. `counts` must be pre-zeroed
/// (use `stream.alloc_zeros`) with length >= max(keys) + 1.
///
/// Atomic adds on u64 — natively supported on Pascal+ (cc 6.0+) which the
/// RTX 5090 satisfies trivially.
pub fn multiplicity_count_by_index_dev(
    keys: &CudaSlice<u64>,
    n: usize,
    counts: &mut CudaSlice<u64>,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    assert!(keys.len() >= n);
    check_grid_bound(n)?;
    if n == 0 {
        return Ok(());
    }
    let be = backend()?;
    let n_u = n as u64;
    unsafe {
        stream
            .launch_builder(&be.multiplicity_count_by_index)
            .arg(keys)
            .arg(&n_u)
            .arg(counts)
            .launch(launch_cfg(n))?;
    }
    Ok(())
}
