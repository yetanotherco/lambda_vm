//! Barycentric evaluation on device — matches
//! `math::polynomial::interpolate_coset_eval_*_with_g_n_inv`.
//!
//! The kernels compute only the unscaled barycentric sum
//!     S = Σ_i point_i * eval_i * inv_denom_i
//! per column. The caller multiplies each `S` by the ext3 scalar
//! `(z^N - g^N) * 1/N * 1/g^N` to get the final OOD value; that scaling is
//! one ext3 mul per column and stays on host.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

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
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }

    let be = backend();
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
    if num_cols == 0 || n == 0 {
        return Ok(vec![0; 3 * num_cols]);
    }

    let be = backend();
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
