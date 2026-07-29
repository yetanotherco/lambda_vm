//! Barycentric evaluation on device. Matches the CPU
//! [`interpolate_coset_eval_ext_with_g_n_inv`](math::polynomial::interpolate_coset_eval_ext_with_g_n_inv)
//!
//! The kernels compute only the unscaled barycentric sum
//!     S = sum over i of point_i * eval_i * inv_denom_i
//! per column. The caller multiplies each `S` by the ext3 scalar
//! `(z^N - g^N) * 1/N * 1/g^N` to get the final OOD value. That scaling is
//! one ext3 mul per column and stays on host.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::lde::{GpuLdeBase, GpuLdeExt3};

const BLOCK_DIM: u32 = 256;

/// Barycentric sums over M base-field columns, each of length `n`, laid out
/// with stride `col_stride` (so column `c` is at `columns[c*col_stride ..
/// c*col_stride + n]`). `inv_denoms` is 3N u64 (ext3 interleaved).
/// Returns 3M u64 (ext3 interleaved), one per column.
pub fn barycentric_base(
    columns: &[u64],
    col_stride: usize,
    coset_points: &[u64],
    inv_denoms_ext3: &[u64],
    n: usize,
    num_cols: usize,
) -> Result<Vec<u64>> {
    assert_eq!(coset_points.len(), n);
    assert_eq!(inv_denoms_ext3.len(), 3 * n);
    assert!(columns.len() >= num_cols * col_stride);
    // Kernel reads col_data[0..n] per column, so col_stride must cover at
    // least n u64s. Smaller strides would read past the column boundary.
    assert!(
        col_stride >= n,
        "col_stride {col_stride} < n {n}: kernel reads col_data[0..n] but stride is shorter"
    );
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }

    let be = backend()?;
    let stream = be.next_stream();

    let cols_dev = stream.clone_htod(&columns[..num_cols * col_stride])?;
    let points_dev = stream.clone_htod(coset_points)?;
    let inv_dev = stream.clone_htod(inv_denoms_ext3)?;
    let mut out_dev = stream.alloc_zeros::<u64>(3 * num_cols)?;

    let col_stride_u64 = col_stride as u64;
    let n_u64 = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (num_cols as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.barycentric_base_batched)
            .arg(&cols_dev)
            .arg(&col_stride_u64)
            .arg(&points_dev)
            .arg(&inv_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Same as [`barycentric_base`] but `columns` holds M ext3 columns in the
/// de-interleaved layout: slab `c*3 + k` at offset `(c*3+k)*col_stride`.
/// `columns.len() >= num_cols * 3 * col_stride`.
pub fn barycentric_ext3(
    columns: &[u64],
    col_stride: usize,
    coset_points: &[u64],
    inv_denoms_ext3: &[u64],
    n: usize,
    num_cols: usize,
) -> Result<Vec<u64>> {
    assert_eq!(coset_points.len(), n);
    assert_eq!(inv_denoms_ext3.len(), 3 * n);
    assert!(columns.len() >= num_cols * 3 * col_stride);
    // Each ext3 slab is read at indices [0..n), so col_stride must cover at
    // least n u64s. Smaller strides would read past the slab boundary.
    assert!(
        col_stride >= n,
        "col_stride {col_stride} < n {n}: kernel reads slab[0..n] but stride is shorter"
    );
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }

    let be = backend()?;
    let stream = be.next_stream();

    let cols_dev = stream.clone_htod(&columns[..num_cols * 3 * col_stride])?;
    let points_dev = stream.clone_htod(coset_points)?;
    let inv_dev = stream.clone_htod(inv_denoms_ext3)?;
    let mut out_dev = stream.alloc_zeros::<u64>(3 * num_cols)?;

    let col_stride_u64 = col_stride as u64;
    let n_u64 = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (num_cols as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.barycentric_ext3_batched)
            .arg(&cols_dev)
            .arg(&col_stride_u64)
            .arg(&points_dev)
            .arg(&inv_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Run `barycentric_base_batched_strided` over the base LDE already on
/// device (`main_handle`), summing over the trace-size coset (every
/// `row_stride = blowup_factor`-th row). H2Ds only the coset points and
/// inv_denoms. The column data never crosses PCIe.
pub fn barycentric_base_on_device(
    main_handle: &GpuLdeBase,
    row_stride: usize,
    coset_points: &[u64],
    inv_denoms_ext3: &[u64],
    n: usize,
) -> Result<Vec<u64>> {
    assert_eq!(coset_points.len(), n);
    assert_eq!(inv_denoms_ext3.len(), 3 * n);
    let num_cols = main_handle.m;
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }
    let col_stride = main_handle.lde_size;

    let be = backend()?;
    let stream = be.next_stream();

    let points_dev = stream.clone_htod(coset_points)?;
    let inv_dev = stream.clone_htod(inv_denoms_ext3)?;
    let mut out_dev = stream.alloc_zeros::<u64>(3 * num_cols)?;

    let col_stride_u64 = col_stride as u64;
    let row_stride_u64 = row_stride as u64;
    let n_u64 = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (num_cols as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.barycentric_base_batched_strided)
            .arg(main_handle.buf.as_ref())
            .arg(&col_stride_u64)
            .arg(&row_stride_u64)
            .arg(&points_dev)
            .arg(&inv_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Same as [`barycentric_base_on_device`] but reads `inv_denoms` AND
/// `coset_points` from device handles (no per-call H2D) and runs on the
/// caller's stream (so the inv_denoms producer and this kernel serialize
/// naturally).
///
/// `inv_denoms_dev` is the full multi-eval-point buffer from
/// `compute_and_invert_denoms_ext3_dev`. `inv_offset_u64` is the start
/// of this eval point's block (in u64s), so the kernel reads
/// `inv_denoms_dev[inv_offset_u64 .. inv_offset_u64 + 3*n]`.
pub fn barycentric_base_on_device_with_dev_inv_denoms(
    stream: &Arc<CudaStream>,
    main_handle: &GpuLdeBase,
    row_stride: usize,
    coset_points_dev: &CudaSlice<u64>,
    inv_denoms_dev: &CudaSlice<u64>,
    inv_offset_u64: usize,
    n: usize,
) -> Result<Vec<u64>> {
    assert!(coset_points_dev.len() >= n);
    let inv_end = inv_offset_u64
        .checked_add(3 * n)
        .expect("barycentric inv_denoms range overflow");
    assert!(inv_end <= inv_denoms_dev.len());
    let num_cols = main_handle.m;
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }
    let col_stride = main_handle.lde_size;

    let be = backend()?;
    let mut out_dev = stream.alloc_zeros::<u64>(3 * num_cols)?;
    let inv_view = inv_denoms_dev.slice(inv_offset_u64..inv_end);
    let points_view = coset_points_dev.slice(0..n);

    let col_stride_u64 = col_stride as u64;
    let row_stride_u64 = row_stride as u64;
    let n_u64 = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (num_cols as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.barycentric_base_batched_strided)
            .arg(main_handle.buf.as_ref())
            .arg(&col_stride_u64)
            .arg(&row_stride_u64)
            .arg(&points_view)
            .arg(&inv_view)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Ext3 counterpart of [`barycentric_base_on_device`]. Reads the aux LDE
/// from the de-interleaved device handle.
pub fn barycentric_ext3_on_device(
    aux_handle: &GpuLdeExt3,
    row_stride: usize,
    coset_points: &[u64],
    inv_denoms_ext3: &[u64],
    n: usize,
) -> Result<Vec<u64>> {
    assert_eq!(coset_points.len(), n);
    assert_eq!(inv_denoms_ext3.len(), 3 * n);
    let num_cols = aux_handle.m;
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }
    let col_stride = aux_handle.lde_size;

    let be = backend()?;
    let stream = be.next_stream();
    // Order this stream against the producer's fill of the LDE (no-op unless the
    // handle carries a ready event, i.e. the composition device-only path).
    if let Some(ev) = aux_handle.ready.as_deref() {
        stream.wait(ev)?;
    }

    let points_dev = stream.clone_htod(coset_points)?;
    let inv_dev = stream.clone_htod(inv_denoms_ext3)?;
    let mut out_dev = stream.alloc_zeros::<u64>(3 * num_cols)?;

    let col_stride_u64 = col_stride as u64;
    let row_stride_u64 = row_stride as u64;
    let n_u64 = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (num_cols as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.barycentric_ext3_batched_strided)
            .arg(aux_handle.buf.as_ref())
            .arg(&col_stride_u64)
            .arg(&row_stride_u64)
            .arg(&points_dev)
            .arg(&inv_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Ext3 counterpart of [`barycentric_base_on_device_with_dev_inv_denoms`].
pub fn barycentric_ext3_on_device_with_dev_inv_denoms(
    stream: &Arc<CudaStream>,
    aux_handle: &GpuLdeExt3,
    row_stride: usize,
    coset_points_dev: &CudaSlice<u64>,
    inv_denoms_dev: &CudaSlice<u64>,
    inv_offset_u64: usize,
    n: usize,
) -> Result<Vec<u64>> {
    assert!(coset_points_dev.len() >= n);
    let inv_end = inv_offset_u64
        .checked_add(3 * n)
        .expect("barycentric inv_denoms range overflow");
    assert!(inv_end <= inv_denoms_dev.len());
    let num_cols = aux_handle.m;
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }
    let col_stride = aux_handle.lde_size;

    let be = backend()?;
    let mut out_dev = stream.alloc_zeros::<u64>(3 * num_cols)?;
    let inv_view = inv_denoms_dev.slice(inv_offset_u64..inv_end);
    let points_view = coset_points_dev.slice(0..n);

    let col_stride_u64 = col_stride as u64;
    let row_stride_u64 = row_stride as u64;
    let n_u64 = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (num_cols as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.barycentric_ext3_batched_strided)
            .arg(aux_handle.buf.as_ref())
            .arg(&col_stride_u64)
            .arg(&row_stride_u64)
            .arg(&points_view)
            .arg(&inv_view)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Gather full rows from a device-resident base-field LDE handle. `rows` are LDE
/// row indices; returns their column values row-major (`rows.len() * main.m`
/// u64, `out[q*num_cols + col]`) — i.e. the concatenation of
/// `gather_main_row(rows[q])` for each `q`. Runs on the caller's stream.
pub fn gather_rows_base_on_device(
    main: &GpuLdeBase,
    rows: &[u32],
    stream: &Arc<CudaStream>,
) -> Result<Vec<u64>> {
    let num_cols = main.m;
    if num_cols == 0 || rows.is_empty() {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let rows_dev = stream.clone_htod(rows)?;
    let mut out = stream.alloc_zeros::<u64>(rows.len() * num_cols)?;
    let col_stride = main.lde_size as u64;
    let num_cols_u64 = num_cols as u64;
    let num_rows_u64 = rows.len() as u64;
    let cfg = LaunchConfig {
        grid_dim: (rows.len() as u32, 1, 1),
        block_dim: (BLOCK_DIM.min(num_cols as u32).max(1), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.gather_rows_base)
            .arg(main.buf.as_ref())
            .arg(&col_stride)
            .arg(&num_cols_u64)
            .arg(&rows_dev)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(cfg)?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Ext3 sibling of [`gather_rows_base_on_device`]: returns `rows.len() * aux.m *
/// 3` u64, interleaved ext3 (`out[(q*num_cols + col)*3 + k]`).
pub fn gather_rows_ext3_on_device(
    aux: &GpuLdeExt3,
    rows: &[u32],
    stream: &Arc<CudaStream>,
) -> Result<Vec<u64>> {
    let num_cols = aux.m;
    if num_cols == 0 || rows.is_empty() {
        return Ok(Vec::new());
    }
    let be = backend()?;
    // Order the caller's stream against the producer's fill (no-op unless the
    // handle carries a ready event, i.e. the composition device-only path).
    if let Some(ev) = aux.ready.as_deref() {
        stream.wait(ev)?;
    }
    let rows_dev = stream.clone_htod(rows)?;
    let mut out = stream.alloc_zeros::<u64>(rows.len() * num_cols * 3)?;
    let col_stride = aux.lde_size as u64;
    let num_cols_u64 = num_cols as u64;
    let num_rows_u64 = rows.len() as u64;
    let cfg = LaunchConfig {
        grid_dim: (rows.len() as u32, 1, 1),
        block_dim: (BLOCK_DIM.min(num_cols as u32).max(1), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.gather_rows_ext3)
            .arg(aux.buf.as_ref())
            .arg(&col_stride)
            .arg(&num_cols_u64)
            .arg(&rows_dev)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(cfg)?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}
