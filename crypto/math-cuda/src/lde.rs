//! Full coset LDE on device. Mirrors `Polynomial::coset_lde_full_expand` in
//! `crypto/math/src/fft/polynomial.rs` algebraically:
//!
//! Input  : N evaluations (natural order) of a poly on the standard subgroup,
//!          plus coset weights (size N). The weights include the `1/N` iFFT
//!          normalisation, matching the `LdeTwiddles::coset_weights` format at
//!          `crypto/stark/src/prover.rs` — i.e. `weights[i] = g^i / N`.
//! Output : N*blowup_factor evaluations (natural order) on the coset.
//!
//! On-device steps, picks a stream from the shared pool so rayon-parallel
//! callers overlap on the GPU. Twiddles are cached in the backend.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{Backend, backend};
use crate::merkle::{keccak_launch_cfg, launch_keccak_base, launch_keccak_base_row_pair};
use crate::ntt::run_ntt_body;

/// Goldilocks `TWO_ADICITY = 32` puts the theoretical domain ceiling at
/// `2^32`, where a downstream `as u32` cast would silently truncate to zero
/// and the corresponding kernel launch would do nothing. Assert at each
/// public entry point before any cast that depends on it.
#[inline]
fn assert_u32_domain(n: usize, what: &str) {
    assert!(
        n <= u32::MAX as usize,
        "{what}: {n} exceeds u32 range — kernel grid would silently truncate",
    );
}

/// Output shape requested from the fused LDE + Keccak entry points.
#[derive(Copy, Clone, PartialEq, Eq)]
enum KeccakCommit {
    /// Only the keccak-256 leaves; no inner-tree build. Caller receives
    /// `num_leaves * 32` bytes.
    LeavesOnly,
    /// Full Merkle tree: leaves at the tail + inner nodes built on-device.
    /// Caller receives `(2*num_leaves - 1) * 32` bytes.
    FullTree,
}

impl KeccakCommit {
    fn total_nodes_bytes(self, num_leaves: usize) -> usize {
        match self {
            KeccakCommit::LeavesOnly => num_leaves * 32,
            KeccakCommit::FullTree => (2 * num_leaves - 1) * 32,
        }
    }

    fn leaves_offset_bytes(self, num_leaves: usize) -> usize {
        match self {
            KeccakCommit::LeavesOnly => 0,
            KeccakCommit::FullTree => (num_leaves - 1) * 32,
        }
    }
}

/// De-interleave `columns` (each `3*n` u64s, ext3-per-element layout
/// `[a, b, c, a, b, c, ...]`) into `pinned` as `3*m` base-field slabs.
/// Component `k` of column `c` lands at `pinned[(c*3 + k)*n .. (c*3 + k)*n + n]`.
///
/// Caller invariants: `pinned.len() >= 3 * columns.len() * n` and each
/// `columns[c].len() >= 3 * n`. The caller must hold the pinned-staging lock.
pub(crate) fn pack_ext3_to_pinned_slabs(columns: &[&[u64]], pinned: &mut [u64], n: usize) {
    let m = columns.len();
    debug_assert!(pinned.len() >= 3 * m * n);
    let pinned_ptr_u = pinned.as_mut_ptr() as usize;
    // Runs under the pinned-staging lock, where rayon can deadlock. See
    // `Backend::pinned_staging`.
    columns.iter().enumerate().for_each(|(c, col)| {
        // SAFETY: each task writes to disjoint `[(c*3 + k)*n .. ..+n]` regions
        // of `pinned`. The outer `&mut [u64]` borrow guarantees no aliasing.
        let slab_a = unsafe {
            std::slice::from_raw_parts_mut((pinned_ptr_u as *mut u64).add((c * 3) * n), n)
        };
        let slab_b = unsafe {
            std::slice::from_raw_parts_mut((pinned_ptr_u as *mut u64).add((c * 3 + 1) * n), n)
        };
        let slab_c = unsafe {
            std::slice::from_raw_parts_mut((pinned_ptr_u as *mut u64).add((c * 3 + 2) * n), n)
        };
        for i in 0..n {
            slab_a[i] = col[i * 3];
            slab_b[i] = col[i * 3 + 1];
            slab_c[i] = col[i * 3 + 2];
        }
    });
}

/// Re-interleave the `3*m` base-field slabs in `pinned` (layout matches
/// `pack_ext3_to_pinned_slabs`) into `outputs`, writing each as
/// `3*lde_size` interleaved u64s.
fn unpack_pinned_slabs_to_ext3(pinned: &[u64], outputs: &mut [&mut [u64]], lde_size: usize) {
    let m = outputs.len();
    debug_assert!(pinned.len() >= 3 * m * lde_size);
    let pinned_const = pinned.as_ptr() as usize;
    // Runs under the pinned-staging lock, where rayon can deadlock. See
    // `Backend::pinned_staging`.
    outputs.iter_mut().enumerate().for_each(|(c, dst)| {
        // SAFETY: each task reads from disjoint `[(c*3 + k)*lde_size .. ..+lde_size]`
        // regions of `pinned`. Caller borrows `pinned` for the duration of the call.
        let slab_a = unsafe {
            std::slice::from_raw_parts(
                (pinned_const as *const u64).add((c * 3) * lde_size),
                lde_size,
            )
        };
        let slab_b = unsafe {
            std::slice::from_raw_parts(
                (pinned_const as *const u64).add((c * 3 + 1) * lde_size),
                lde_size,
            )
        };
        let slab_c = unsafe {
            std::slice::from_raw_parts(
                (pinned_const as *const u64).add((c * 3 + 2) * lde_size),
                lde_size,
            )
        };
        for i in 0..lde_size {
            dst[i * 3] = slab_a[i];
            dst[i * 3 + 1] = slab_b[i];
            dst[i * 3 + 2] = slab_c[i];
        }
    });
}

/// Run `bit_reverse_permute_batched` over `m` columns of length `n` each
/// (column stride `col_stride`). 256 threads per block, grid sized to cover
/// `n` per column.
fn launch_bit_reverse_batched(
    stream: &CudaStream,
    be: &Backend,
    buf: &mut CudaSlice<u64>,
    n: u64,
    log_n: u64,
    col_stride: u64,
    m: u32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), m, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute_batched)
            .arg(buf)
            .arg(&n)
            .arg(&log_n)
            .arg(&col_stride)
            .launch(cfg)?;
    }
    Ok(())
}

/// D2H `dst.len()` bytes from `dev_bytes` into the caller's pageable `dst`
/// via the pinned-hashes staging buffer. Synchronises the stream first (so
/// any other D2H queued on the same stream also drains), then does a rayon
/// chunked memcpy pinned → caller to spread page-fault cost across cores.
fn d2h_bytes_via_pinned_hashes(
    stream: &Arc<CudaStream>,
    be: &Backend,
    dev_bytes: &CudaSlice<u8>,
    dst: &mut [u8],
) -> Result<()> {
    let pending =
        crate::device::async_dtoh_via(stream, be.pinned_hashes(), &be.ctx, dev_bytes, dst.len())?;
    // Waits only for work queued up to the copy (event), not the whole stream.
    pending.wait_into_bytes(dst)
}

/// Run `pointwise_mul_batched`: `buf[c*col_stride + i] *= weights[i]` for
/// `m` columns, `n` elements each.
fn launch_pointwise_mul_batched(
    stream: &CudaStream,
    be: &Backend,
    buf: &mut CudaSlice<u64>,
    weights: &CudaSlice<u64>,
    n: u64,
    col_stride: u64,
    m: u32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), m, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.pointwise_mul_batched)
            .arg(buf)
            .arg(weights)
            .arg(&n)
            .arg(&col_stride)
            .launch(cfg)?;
    }
    Ok(())
}

// ── Row-major NTT helpers ────────────────────────────────────────────────────

fn launch_bit_reverse_row_major(
    stream: &CudaStream,
    be: &Backend,
    buf: &mut CudaSlice<u64>,
    n: u64,
    log_n: u64,
    m: u64,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: ((m as u32).div_ceil(256), (n as u32).min(65535), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_row_major)
            .arg(buf)
            .arg(&n)
            .arg(&log_n)
            .arg(&m)
            .launch(cfg)?;
    }
    Ok(())
}

fn launch_pointwise_mul_row_major(
    stream: &CudaStream,
    be: &Backend,
    buf: &mut CudaSlice<u64>,
    weights: &CudaSlice<u64>,
    n: u64,
    m: u64,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: ((m as u32).div_ceil(256), (n as u32).min(65535), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.pointwise_mul_row_major)
            .arg(buf)
            .arg(weights)
            .arg(&n)
            .arg(&m)
            .launch(cfg)?;
    }
    Ok(())
}

fn run_row_major_ntt_body(
    stream: &CudaStream,
    be: &Backend,
    buf: &mut CudaSlice<u64>,
    tw: &CudaSlice<u64>,
    n: u64,
    log_n: u64,
    m: u64,
) -> Result<()> {
    let col_tile: u32 = 32.min(m as u32);
    let row_tile: u32 = (256 / col_tile).max(1);
    for level in 0..log_n {
        let cfg = LaunchConfig {
            grid_dim: (
                (m as u32).div_ceil(col_tile),
                ((n >> 1) as u32).div_ceil(row_tile).min(65535),
                1,
            ),
            block_dim: (col_tile, row_tile, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_level_row_major)
                .arg(&mut *buf)
                .arg(tw)
                .arg(&n)
                .arg(&log_n)
                .arg(&level)
                .arg(&m)
                .launch(cfg)?;
        }
    }
    Ok(())
}

/// Row-major ROW-PAIR leaf hashing: leaf `i` hashes the two consecutive
/// bit-reversed rows `reverse_index(2i)`, `reverse_index(2i+1)` (each `m` lanes,
/// read contiguously from the row-major `buf`), producing `num_rows / 2` leaves.
/// Row-major analog of [`launch_keccak_base_row_pair`]; matches the CPU
/// `commit_bit_reversed(.., 2)` and the verifier's `verify_opening_pair`.
fn launch_keccak_base_row_major_row_pair(
    stream: &CudaStream,
    be: &Backend,
    buf: &CudaSlice<u64>,
    m: u64,
    num_rows: u64,
    log_num_rows: u64,
    leaves_out: &mut cudarc::driver::CudaViewMut<'_, u8>,
) -> Result<()> {
    // Register-heavy Keccak kernel: launch with the keccak-tuned block dim (128,
    // via `keccak_launch_cfg`); a larger block exceeds the per-block register
    // budget and fails the launch (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES). The kernel
    // derives rows as `__brevll(2*tid + k) >> (64 - log_num_rows)`; a 64-bit shift
    // is UB at `log_num_rows == 0`, so require `num_rows >= 2` (also the minimum
    // for a single row pair).
    debug_assert!(
        num_rows >= 2,
        "row-major row-pair keccak requires num_rows >= 2"
    );
    // One thread per leaf (= one bit-reversed row pair).
    let cfg = keccak_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.keccak256_leaves_base_row_major_row_pair)
            .arg(buf)
            .arg(&m)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves_out)
            .launch(cfg)?;
    }
    Ok(())
}

/// Column-range variant of [`launch_keccak_base_row_major_row_pair`]: leaves
/// hash only columns `[col_start, col_end)` of each bit-reversed row pair
/// (`m` stays the full row stride). Matches the CPU
/// `commit_rows_bit_reversed_subset`.
#[allow(clippy::too_many_arguments)]
fn launch_keccak_base_row_major_row_pair_range(
    stream: &CudaStream,
    be: &Backend,
    buf: &CudaSlice<u64>,
    m: u64,
    col_start: u64,
    col_end: u64,
    num_rows: u64,
    log_num_rows: u64,
    leaves_out: &mut cudarc::driver::CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(
        num_rows >= 2,
        "row-major row-pair keccak requires num_rows >= 2"
    );
    debug_assert!(
        col_start < col_end && col_end <= m,
        "column range in bounds"
    );
    let cfg = keccak_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.keccak256_leaves_base_row_major_row_pair_range)
            .arg(buf)
            .arg(&m)
            .arg(&col_start)
            .arg(&col_end)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves_out)
            .launch(cfg)?;
    }
    Ok(())
}

/// Transpose row-major `lde_size × cols` → column-major with stride `lde_size`,
/// returning the new device buffer. Used to convert the row-major LDE output to
/// the column-major layout expected by downstream GPU kernels (DEEP, barycentric).
/// No synchronize — callers on the same stream are ordered; other streams must
/// synchronize themselves.
fn launch_row_to_col_major(
    stream: &Arc<CudaStream>,
    be: &Backend,
    src: &CudaSlice<u64>,
    lde_size: usize,
    cols: usize,
    lde_u64: u64,
) -> Result<CudaSlice<u64>> {
    let mut dst = stream.alloc_zeros::<u64>(lde_size * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (
            (cols as u32).div_ceil(32),
            (lde_size as u32).div_ceil(32).min(65535),
            1,
        ),
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.matrix_transpose_strided)
            .arg(src)
            .arg(&mut dst)
            .arg(&(lde_size as u32))
            .arg(&(cols as u32))
            .arg(&lde_u64)
            .launch(cfg)?;
    }
    Ok(dst)
}

/// Row-major LDE input: either a host slice (uploaded) or an already-resident
/// device buffer (copied device-to-device, no PCIe upload).
enum InnerInput<'a> {
    Host(&'a [u64]),
    Dev(&'a CudaSlice<u64>),
}

/// The expansion stage shared by the row-major commit pipelines: upload (or
/// D2D-copy) the row-major trace into a zero-padded `lde_size × total_cols`
/// buffer, optionally snapshot the trace-domain input column-major (for the
/// LogUp fingerprint kernel), then iNTT → coset weights → forward NTT in
/// place. Returns the row-major LDE buffer and the optional snapshot.
#[allow(clippy::too_many_arguments)]
fn expand_row_major_on_stream(
    stream: &Arc<CudaStream>,
    be: &Backend,
    input: InnerInput,
    n: usize,
    total_cols: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_trace_col_major: bool,
) -> Result<(CudaSlice<u64>, Option<CudaSlice<u64>>)> {
    let lde_size = n * blowup_factor;
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;
    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let cols_u64 = total_cols as u64;

    // Fill a zeroed lde_size*total_cols buffer; only the first n*total_cols rows
    // carry data, the remainder are already zero (zero-padding for LDE). Host
    // input uploads (H2D); device input copies in place (D2D, no PCIe upload).
    let mut buf = stream.alloc_zeros::<u64>(lde_size * total_cols)?;
    {
        match input {
            InnerInput::Host(h) => stream.memcpy_htod(h, &mut buf.slice_mut(0..n * total_cols))?,
            InnerInput::Dev(d) => stream.memcpy_dtod(d, &mut buf.slice_mut(0..n * total_cols))?,
        }
    }

    // Snapshot the trace-domain input (column-major) before the iNTT overwrites
    // it in place. The LogUp aux fingerprint kernel reads the main trace in
    // place from this buffer, so R1 aux build skips the ~3 GB main re-upload.
    // Transpose is a plain row->col transpose on the first n rows (not yet
    // bit-reversed): dst[col*n + row] = buf[row*total_cols + col].
    let trace_col_major = if retain_trace_col_major {
        Some(launch_row_to_col_major(
            stream, be, &buf, n, total_cols, n as u64,
        )?)
    } else {
        None
    };

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    // iNTT: bit-reverse rows → per-level DIT.
    launch_bit_reverse_row_major(stream.as_ref(), be, &mut buf, n_u64, log_n, cols_u64)?;
    run_row_major_ntt_body(
        stream.as_ref(),
        be,
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        cols_u64,
    )?;

    // Coset weights: one weight per row, broadcast across all columns.
    launch_pointwise_mul_row_major(stream.as_ref(), be, &mut buf, &weights_dev, n_u64, cols_u64)?;

    // Forward NTT at lde_size.
    launch_bit_reverse_row_major(stream.as_ref(), be, &mut buf, lde_u64, log_lde, cols_u64)?;
    run_row_major_ntt_body(
        stream.as_ref(),
        be,
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        cols_u64,
    )?;

    Ok((buf, trace_col_major))
}

/// Shared row-major LDE + Keccak + Merkle pipeline for the base and ext3 paths.
///
/// `total_cols` is the number of base-field columns in the row-major layout:
/// `m` for base, `m * 3` for ext3. Because `Fp3 = [u64; 3]`, the three ext3
/// components are just three adjacent base-field columns, so the same row-major
/// NTT and Keccak kernels process all of them simultaneously — no de-interleave.
///
/// Single H2D (or D2D), row-major NTT, single D2H — no CPU-side extract or
/// transpose. Returns (merkle_nodes, column-major device buffer, row-major LDE
/// Vec, optional trace-domain column-major snapshot — `Some` iff
/// `retain_trace_col_major`). The buffer is transposed to column-major (as
/// required by the downstream GPU kernels DEEP/barycentric); callers wrap it in
/// the appropriate LDE handle.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn coset_lde_row_major_inner(
    input: InnerInput,
    n: usize,
    total_cols: usize,
    blowup_factor: usize,
    weights: &[u64],
    what: &str,
    retain_trace_col_major: bool,
    retain_host_lde: bool,
) -> Result<(
    GpuMerkleTree,
    CudaSlice<u64>,
    Vec<u64>,
    Option<CudaSlice<u64>>,
    Arc<crate::device::PooledEvent>,
)> {
    let input_len = match &input {
        InnerInput::Host(h) => h.len(),
        InnerInput::Dev(d) => d.len(),
    };
    assert_eq!(input_len, n * total_cols);
    assert!(n.is_power_of_two());
    assert_eq!(weights.len(), n);
    assert!(blowup_factor.is_power_of_two());
    let lde_size = n * blowup_factor;
    assert_u32_domain(lde_size, what);

    // Row-pair trace commit: one Merkle leaf per bit-reversed row pair (rows 2i,
    // 2i+1), matching the CPU `commit_bit_reversed(.., ROWS_PER_LEAF=2)` and the
    // verifier's `verify_opening_pair`. `lde_size` is a power of two >= 2, so it
    // is always even.
    let num_leaves = lde_size / 2;
    let nodes_bytes = KeccakCommit::FullTree.total_nodes_bytes(num_leaves);
    let log_lde = lde_size.trailing_zeros() as u64;
    let lde_u64 = lde_size as u64;
    let cols_u64 = total_cols as u64;

    let be = backend()?;
    let stream = be.next_stream();

    let (buf, trace_col_major) = expand_row_major_on_stream(
        &stream,
        be,
        input,
        n,
        total_cols,
        blowup_factor,
        weights,
        retain_trace_col_major,
    )?;

    // Keccak + Merkle on-device. Each row-pair leaf reads two bit-reversed rows
    // of `total_cols` consecutive u64s (`lde_u64` is the bit-reverse modulus; the
    // kernel emits `lde_size / 2` leaves).
    let mut nodes_dev = unsafe { stream.alloc::<u8>(nodes_bytes) }?;
    let leaves_offset = KeccakCommit::FullTree.leaves_offset_bytes(num_leaves);
    {
        let mut leaves_view = nodes_dev.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
        launch_keccak_base_row_major_row_pair(
            stream.as_ref(),
            be,
            &buf,
            cols_u64,
            lde_u64,
            log_lde,
            &mut leaves_view,
        )?;
    }
    crate::merkle::build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;

    // Copy the 32-byte root BEFORE queueing the big drain/transpose: this
    // pageable copy host-blocks until everything queued so far lands, so
    // keeping it early means it waits for the tree kernels only (the root is
    // needed now regardless — Fiat-Shamir absorbs it before anything else).
    // The Merkle tree stays resident on device; query openings gather paths
    // from it (see merkle::gather_merkle_paths_dev).
    let mut root = [0u8; 32];
    stream.memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;

    // D2H the row-major LDE (skipped when `retain_host_lde` is false — the
    // full-residency path keeps the LDE device-only; that skip is the big
    // transfer/alloc win, and we return an empty host Vec).
    let lde_pending = if retain_host_lde {
        Some(crate::device::async_dtoh_via(
            &stream,
            be.pinned_staging(),
            &be.ctx,
            &buf,
            lde_size * total_cols,
        )?)
    } else {
        None
    };

    // Transpose row-major buf into column-major for the handle. Downstream
    // kernels (DEEP, barycentric) expect buf[c * lde_size + r] (column-major).
    let col_major_dev = launch_row_to_col_major(&stream, be, &buf, lde_size, total_cols, lde_u64)?;
    // No host synchronize here: the handle carries a `ready` event instead,
    // and consumers on other streams wait on it device-side
    // (`wait_ready_on`). On the device-only path this makes the whole
    // commit's tail (transpose) run behind the host's next work.
    let ready = be.take_event()?;
    ready.event().record(&stream)?;
    let lde_out = match lde_pending {
        Some(p) => {
            let mut out = vec![0u64; lde_size * total_cols];
            p.wait_into_u64(&mut out)?;
            out
        }
        None => Vec::new(),
    };

    let tree = GpuMerkleTree {
        nodes: Arc::new(nodes_dev),
        leaves_len: num_leaves,
        root,
    };
    Ok((
        tree,
        col_major_dev,
        lde_out,
        trace_col_major,
        Arc::new(ready),
    ))
}

/// Row-major LDE + Keccak + Merkle, all on-device, keeping the Merkle tree
/// resident on device (in the handle's `tree`). The host tree is not built, so
/// the whole tree copy to host is eliminated; query openings gather paths from
/// the device tree.
///
/// Input: `row_major` is a flat `n * m` slice in row-major order. Returns the
/// `GpuLdeBase` handle (column-major buf, plus the device tree) and the
/// row-major LDE Vec.
pub fn coset_lde_row_major_with_merkle_tree_keep(
    row_major: &[u64],
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_host_lde: bool,
) -> Result<(GpuLdeBase, Vec<u64>)> {
    let (tree, col_major_dev, lde_out, trace_col_major, ready) = coset_lde_row_major_inner(
        InnerInput::Host(row_major),
        n,
        m,
        blowup_factor,
        weights,
        "coset_lde_row_major lde_size",
        true,
        retain_host_lde,
    )?;
    let handle = GpuLdeBase {
        buf: Arc::new(col_major_dev),
        m,
        lde_size: n * blowup_factor,
        tree: Some(tree),
        ready: Some(ready),
        trace_dev: trace_col_major.map(Arc::new),
        trace_rows: n,
    };
    Ok((handle, lde_out))
}

/// Row-major LDE + TWO subset Merkle trees for preprocessed tables: the
/// precomputed columns `[0, split_col)` and the multiplicity columns
/// `[split_col, m)` commit to separate trees over the same row-major LDE,
/// mirroring the CPU `commit_rows_bit_reversed_subset` pair.
///
/// Both trees' complete node buffers are downloaded to host
/// (`(2*num_leaves - 1) * 32` bytes each, inner nodes first, root at offset 0,
/// leaves at the tail — the exact `MerkleTree::from_precomputed_nodes`
/// layout), because preprocessed-table openings walk host trees. The
/// precomputed tree is only built when `build_precomputed` is true (the
/// caller skips it on a process-cache hit).
///
/// Returns `(precomputed_nodes, mult_nodes, handle, row_major_lde)`. The
/// handle carries the column-major LDE + trace snapshot for downstream GPU
/// rounds but NO device tree (`tree: None`) — openings for preprocessed
/// tables never gather from device.
#[allow(clippy::type_complexity)]
pub fn coset_lde_row_major_split_trees(
    row_major: &[u64],
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    split_col: usize,
    build_precomputed: bool,
) -> Result<(Option<Vec<u8>>, Vec<u8>, GpuLdeBase, Vec<u64>)> {
    assert!(split_col > 0 && split_col < m, "split inside the row");
    let lde_size = n * blowup_factor;
    assert_u32_domain(lde_size, "coset_lde_row_major_split lde_size");
    let num_leaves = lde_size / 2;
    let nodes_bytes = KeccakCommit::FullTree.total_nodes_bytes(num_leaves);
    let leaves_offset = KeccakCommit::FullTree.leaves_offset_bytes(num_leaves);
    let log_lde = lde_size.trailing_zeros() as u64;
    let lde_u64 = lde_size as u64;
    let cols_u64 = m as u64;

    let be = backend()?;
    let stream = be.next_stream();

    let (buf, trace_col_major) = expand_row_major_on_stream(
        &stream,
        be,
        InnerInput::Host(row_major),
        n,
        m,
        blowup_factor,
        weights,
        true,
    )?;

    // One subset tree per column range, built sequentially on the stream.
    let build_subset_tree = |col_start: u64, col_end: u64| -> Result<Vec<u8>> {
        let mut nodes_dev = unsafe { stream.alloc::<u8>(nodes_bytes) }?;
        {
            let mut leaves_view =
                nodes_dev.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
            launch_keccak_base_row_major_row_pair_range(
                stream.as_ref(),
                be,
                &buf,
                cols_u64,
                col_start,
                col_end,
                lde_u64,
                log_lde,
                &mut leaves_view,
            )?;
        }
        crate::merkle::build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
        let mut nodes_host = vec![0u8; nodes_bytes];
        {
            stream.memcpy_dtoh(&nodes_dev, &mut nodes_host)?;
        }
        Ok(nodes_host)
    };

    let precomputed_nodes = if build_precomputed {
        Some(build_subset_tree(0, split_col as u64)?)
    } else {
        None
    };
    let mult_nodes = build_subset_tree(split_col as u64, cols_u64)?;

    // D2H the row-major LDE (preprocessed tables always keep the host copy —
    // they are excluded from the device-only gate).
    let lde_pending =
        crate::device::async_dtoh_via(&stream, be.pinned_staging(), &be.ctx, &buf, lde_size * m)?;

    // Column-major handle for downstream GPU rounds (DEEP, barycentric,
    // constraint composition).
    let col_major_dev = launch_row_to_col_major(&stream, be, &buf, lde_size, m, lde_u64)?;
    let ready = be.take_event()?;
    ready.event().record(&stream)?;

    let lde_out = {
        let mut out = vec![0u64; lde_size * m];
        lde_pending.wait_into_u64(&mut out)?;
        out
    };

    let handle = GpuLdeBase {
        buf: Arc::new(col_major_dev),
        m,
        lde_size,
        tree: None,
        ready: Some(Arc::new(ready)),
        trace_dev: trace_col_major.map(Arc::new),
        trace_rows: n,
    };
    Ok((precomputed_nodes, mult_nodes, handle, lde_out))
}

/// Row-major ext3 LDE + Keccak + Merkle, all on-device.
///
/// `Fp3` is `[u64; 3]` in memory, so row-major ext3 with `m` ext3 columns is
/// identical to row-major base-field with `m3 = m * 3`. The same row-major NTT
/// and Keccak kernels handle all three components simultaneously — no extra
/// de-interleave step.
///
/// Input: `row_major` is `n * m` ext3 elements as flat `n * m * 3` u64s
/// (element [row][col] components k=0,1,2 at `row_major[(row*m + col)*3 + k]`).
/// Returns (merkle_nodes, GpuLdeExt3 handle, row-major ext3 LDE Vec<u64>).
pub fn coset_lde_ext3_row_major_with_merkle_tree_keep(
    row_major: &[u64],
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_host_lde: bool,
) -> Result<(GpuLdeExt3, Vec<u64>)> {
    let (tree, col_major_dev, lde_out, _, ready) = coset_lde_row_major_inner(
        InnerInput::Host(row_major),
        n,
        m * 3,
        blowup_factor,
        weights,
        "coset_lde_ext3_row_major lde_size",
        false,
        retain_host_lde,
    )?;
    let handle = GpuLdeExt3 {
        buf: Arc::new(col_major_dev),
        m,
        lde_size: n * blowup_factor,
        tree: Some(tree),
        ready: Some(ready),
    };
    Ok((handle, lde_out))
}

/// Like [`coset_lde_ext3_row_major_with_merkle_tree_keep`] but the input is an
/// already-resident device buffer (`n * m` ext3 elements, row-major, `n*m*3`
/// u64s). No PCIe upload: the buffer is copied device-to-device into the LDE
/// scratch. Used by the resident LogUp aux path.
pub fn coset_lde_ext3_row_major_with_merkle_tree_keep_dev(
    input_dev: &CudaSlice<u64>,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_host_lde: bool,
) -> Result<(GpuLdeExt3, Vec<u64>)> {
    let (tree, col_major_dev, lde_out, _, ready) = coset_lde_row_major_inner(
        InnerInput::Dev(input_dev),
        n,
        m * 3,
        blowup_factor,
        weights,
        "coset_lde_ext3_row_major_dev lde_size",
        false,
        retain_host_lde,
    )?;
    let handle = GpuLdeExt3 {
        buf: Arc::new(col_major_dev),
        m,
        lde_size: n * blowup_factor,
        tree: Some(tree),
        ready: Some(ready),
    };
    Ok((handle, lde_out))
}

/// Handle to a base-field LDE kept live on device after R1 commit.
/// Layout: `m` columns, each `lde_size` u64s, column `c` at byte offset
/// `c * lde_size * 8` within `buf`. Freed when `buf` Arc drops.
///
/// `tree` optionally carries the main trace Merkle tree kept resident on device
/// (the keep path), so R4 query openings gather paths on device instead of
/// copying the whole tree to host. None on the CPU path.
#[derive(Clone)]
pub struct GpuLdeBase {
    pub buf: Arc<CudaSlice<u64>>,
    pub m: usize,
    pub lde_size: usize,
    pub tree: Option<GpuMerkleTree>,
    /// Trace-domain main columns, column-major `[col*trace_rows + row]`, kept
    /// resident from the R1 main LDE so the LogUp aux fingerprint kernel reads
    /// them in place (no re-upload). None unless the base keep path retained it.
    pub trace_dev: Option<Arc<CudaSlice<u64>>>,
    /// Row count (n) of `trace_dev`; 0 when `trace_dev` is None.
    pub trace_rows: usize,
    /// Fires once `buf` is fully written (recorded after the producer's last
    /// kernel). `None` means the producer synchronized before returning.
    /// Consumers on other streams call [`GpuLdeBase::wait_ready_on`].
    pub ready: Option<Arc<crate::device::PooledEvent>>,
}

impl GpuLdeBase {
    /// Make `stream` wait (device-side, no host block) until `buf` is ready.
    pub fn wait_ready_on(&self, stream: &CudaStream) -> Result<()> {
        match &self.ready {
            Some(ev) => stream.wait(ev.event()),
            None => Ok(()),
        }
    }
}

/// Handle to an ext3 LDE kept live on device, de-interleaved into 3 base
/// slabs per column. Column `c` component `k` at u64 offset
/// `(c*3 + k) * lde_size` within `buf`.
#[derive(Clone)]
pub struct GpuLdeExt3 {
    pub buf: Arc<CudaSlice<u64>>,
    pub m: usize,
    pub lde_size: usize,
    /// Optionally the aux or composition Merkle tree kept resident on device
    /// (the keep path), so R4 openings gather paths on device. None otherwise.
    pub tree: Option<GpuMerkleTree>,
    /// Fires once `buf` is fully written. `None` = producer synchronized.
    /// Consumers on other streams call [`GpuLdeExt3::wait_ready_on`].
    pub ready: Option<Arc<crate::device::PooledEvent>>,
}

impl GpuLdeExt3 {
    /// Make `stream` wait (device-side, no host block) until `buf` is ready.
    pub fn wait_ready_on(&self, stream: &CudaStream) -> Result<()> {
        match &self.ready {
            Some(ev) => stream.wait(ev.event()),
            None => Ok(()),
        }
    }
}

/// Merkle tree kept resident on device after a commit, so query openings gather
/// paths on device instead of copying the whole tree to host. Node layout
/// matches the CPU tree (`crypto/crypto/src/merkle_tree`): `nodes[0..leaves_len-1]`
/// are inner nodes (root at 0), `nodes[leaves_len-1..]` are the leaves, each 32
/// bytes. Freed when the `nodes` Arc drops.
#[derive(Clone)]
pub struct GpuMerkleTree {
    pub nodes: Arc<CudaSlice<u8>>,
    pub leaves_len: usize,
    /// The Merkle root (node 0), copied to host at build time so the commitment
    /// is available without copying the whole tree.
    pub root: [u8; 32],
}

pub fn coset_lde_base(evals: &[u64], blowup_factor: usize, weights: &[u64]) -> Result<Vec<u64>> {
    let n = evals.len();
    // Empty input must short-circuit before the power-of-two assert
    // (is_power_of_two returns false for 0).
    if n == 0 {
        return Ok(Vec::new());
    }
    assert!(n.is_power_of_two(), "evals length must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match evals");
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup must be power of two"
    );
    let lde_size = n * blowup_factor;
    assert_u32_domain(lde_size, "coset_lde_base lde_size");
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend()?;
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
    {
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute)
                .arg(&mut buf)
                .arg(&n_u64)
                .arg(&log_n)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // Note: `run_ntt_body` expects a standalone CudaSlice; we pass `buf` and
    // the kernel walks the first `n_u64` elements via its own indexing.
    run_ntt_body(stream.as_ref(), &mut buf, inv_tw.as_ref(), n_u64, log_n)?;
    // Note: the CPU iFFT does not include 1/N — it's folded into `weights`. The
    // next pointwise multiply applies both the coset shift and the 1/N factor.

    // === 2. Pointwise multiply first N by coset weights (includes 1/N) ===
    {
        unsafe {
            stream
                .launch_builder(&be.pointwise_mul)
                .arg(&mut buf)
                .arg(&weights_dev)
                .arg(&n_u64)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // === 3. Forward NTT on full buffer ===
    {
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute)
                .arg(&mut buf)
                .arg(&lde_u64)
                .arg(&log_lde)
                .launch(LaunchConfig::for_num_elems(lde_size as u32))?;
        }
    }
    run_ntt_body(stream.as_ref(), &mut buf, fwd_tw.as_ref(), lde_u64, log_lde)?;

    let out = { stream.clone_dtoh(&buf)? };
    {
        stream.synchronize()?;
    }
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
    // Empty columns must short-circuit before the power-of-two assert
    // (is_power_of_two returns false for 0).
    if n == 0 {
        return Ok(vec![Vec::new(); m]);
    }
    assert!(n.is_power_of_two(), "column length must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match column length");
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup must be power of two"
    );
    for c in columns.iter() {
        assert_eq!(c.len(), n, "all columns must be the same size");
    }
    let lde_size = n * blowup_factor;
    assert_u32_domain(lde_size, "coset_lde_batch_base lde_size");
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    // Pinned staging. Lock and grow to max(m*n for upload, m*lde_size for
    // download). The guard is held from the pack until the async uploads have
    // landed (the H2D DMA reads the slab directly); the D2H drain at the end
    // re-acquires the slot via `async_dtoh_via`.
    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(m * lde_size, &be.ctx)?;
    // SAFETY: staging is locked, the slice alias ends before we unlock.
    let pinned = unsafe { staging.as_mut_slice(m * lde_size) };

    // Pack columns into the first m*n slots of the pinned buffer. Runs under
    // the pinned-staging lock, where rayon can deadlock. See
    // `Backend::pinned_staging`.
    {
        for (c, col) in columns.iter().enumerate() {
            pinned[c * n..c * n + n].copy_from_slice(col);
        }
    }

    // Column layout: `buf[c * lde_size + r]`. Zeroed so the [n, lde_size)
    // tail of each column is already the zero-pad the CPU path does.
    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    // One memcpy per column from the pinned buffer into the strided slots.
    // The pinned source hits PCIe line-rate.
    {
        for c in 0..m {
            let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
            stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
        }
    }
    // The uploads above are truly asynchronous (pinned source), so the
    // staging slot must stay locked until they land; the slot's reusable event marks that
    // point. It is waited just before the D2H drain re-acquires the slot.
    staging.record_event(&stream)?;

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let m_u32 = m as u32;

    // === 1. Bit-reverse first N of every column ===
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;

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

    // === 3. Pointwise multiply by coset weights (includes 1/N) ===
    launch_pointwise_mul_batched(
        stream.as_ref(),
        be,
        &mut buf,
        &weights_dev,
        n_u64,
        col_stride_u64,
        m_u32,
    )?;

    // === 4. Bit-reverse full LDE of every column ===
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;

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

    // Release the staging slot before the drain: the uploads have landed once
    // the slot event fires (the NTT kernels above are queued behind them, so the
    // GPU stays busy while the host waits here).
    staging.sync_event()?;
    drop(staging);

    // Single big D2H into the reusable pinned staging buffer — pinned, one
    // call to the driver, saturates PCIe. Enqueued without blocking; the host
    // blocks once, in `wait_and_read` below.
    let pending =
        crate::device::async_dtoh_via(&stream, staging_slot, &be.ctx, &buf, m * lde_size)?;

    // Split pinned into per-column Vec<u64>s. Runs under the pinned-staging
    // lock (held by `pending`), where rayon can deadlock. See
    // `Backend::pinned_staging`.
    let out: Vec<Vec<u64>> = {
        pending.wait_and_read(|bytes| {
            // SAFETY: the pinned slab is u64-aligned by construction and the
            // copy deposited exactly `m * lde_size` u64s.
            let pinned =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, m * lde_size) };
            (0..m)
                .map(|c| {
                    // set_len skips the O(N) zero-init that vec![0; n] would
                    // do. copy_from_slice below writes every slot before any
                    // reader sees the Vec.
                    #[allow(clippy::uninit_vec)]
                    let mut v = {
                        let mut v = Vec::<u64>::with_capacity(lde_size);
                        unsafe { v.set_len(lde_size) };
                        v
                    };
                    v.copy_from_slice(&pinned[c * lde_size..c * lde_size + lde_size]);
                    v
                })
                .collect()
        })?
    };
    Ok(out)
}

/// Like `coset_lde_batch_base` but writes directly into caller-provided
/// output slices instead of allocating fresh `Vec<u64>`s. Each output slice
/// must already have length `n * blowup_factor`. Avoids pageable allocator
/// work and page faults at prover scale because the caller's Vecs have been
/// sized once and are reused across calls.
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
    // Empty columns must short-circuit before the power-of-two assert
    // (is_power_of_two returns false for 0).
    if n == 0 {
        return Ok(());
    }
    assert!(n.is_power_of_two(), "column length must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match column length");
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup must be power of two"
    );
    for c in columns.iter() {
        assert_eq!(c.len(), n, "all columns must be the same size");
    }
    let lde_size = n * blowup_factor;
    for o in outputs.iter() {
        assert_eq!(o.len(), lde_size, "each output must be lde_size");
    }
    assert_u32_domain(lde_size, "coset_lde_batch_base_into lde_size");
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(m * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(m * lde_size) };

    {
        for (c, col) in columns.iter().enumerate() {
            pinned[c * n..c * n + n].copy_from_slice(col);
        }
    }

    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    {
        for c in 0..m {
            let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
            stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
        }
    }
    // The uploads above are truly asynchronous (pinned source); the staging
    // slot stays locked until this event fires (waited before the drain).
    staging.record_event(&stream)?;

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let m_u32 = m as u32;

    // iNTT bit-reverse + body, pointwise mul, forward bit-reverse + body.
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;
    launch_pointwise_mul_batched(
        stream.as_ref(),
        be,
        &mut buf,
        &weights_dev,
        n_u64,
        col_stride_u64,
        m_u32,
    )?;
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;

    // Release the staging slot before the drain: the uploads have landed once
    // the slot event fires (the kernels above are queued behind them).
    staging.sync_event()?;
    drop(staging);

    // Big D2H enqueued without blocking; the host blocks once, in
    // `wait_and_read` below.
    let pending =
        crate::device::async_dtoh_via(&stream, staging_slot, &be.ctx, &buf, m * lde_size)?;

    // Copy pinned into caller outputs. Runs under the pinned-staging lock
    // (held by `pending`), where rayon can deadlock. See
    // `Backend::pinned_staging`.
    {
        pending.wait_and_read(|bytes| {
            // SAFETY: the pinned slab is u64-aligned by construction and the
            // copy deposited exactly `m * lde_size` u64s.
            let pinned =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, m * lde_size) };
            for (c, dst) in outputs.iter_mut().enumerate() {
                dst.copy_from_slice(&pinned[c * lde_size..c * lde_size + lde_size]);
            }
        })?;
    }
    Ok(())
}

/// Fused LDE + row-pair Keccak-256 leaf hashing. Caller receives
/// `(lde_size / 2) * 32` bytes of leaf hashes in `hashed_leaves_out` (one
/// 32-byte digest per bit-reversed row pair, in natural leaf order, matching
/// `commit_bit_reversed(.., 2)` on the CPU side). Thin wrapper over
/// `coset_lde_batch_base_into_with_merkle_tree_inner` with `LeavesOnly` — no
/// inner-tree build, no device handle.
pub fn coset_lde_batch_base_into_with_leaf_hash(
    columns: &[&[u64]],
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    hashed_leaves_out: &mut [u8],
) -> Result<()> {
    coset_lde_batch_base_into_with_merkle_tree_inner(
        columns,
        blowup_factor,
        weights,
        outputs,
        hashed_leaves_out,
        KeccakCommit::LeavesOnly,
        false,
        2,
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn coset_lde_batch_base_into_with_merkle_tree_inner(
    columns: &[&[u64]],
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    nodes_out: &mut [u8],
    commit: KeccakCommit,
    keep_device_buf: bool,
    // 1 = one leaf per bit-reversed row; 2 = one leaf per row pair (2i, 2i+1),
    // matching the CPU `commit_bit_reversed(.., 2)` used for the trace commit.
    rows_per_leaf: usize,
) -> Result<Option<GpuLdeBase>> {
    if columns.is_empty() {
        assert_eq!(outputs.len(), 0);
        return Ok(None);
    }
    let m = columns.len();
    assert_eq!(outputs.len(), m);
    let n = columns[0].len();
    // (is_power_of_two returns false for 0).
    if n == 0 {
        return Ok(None);
    }
    assert!(n.is_power_of_two());
    assert_eq!(weights.len(), n);
    assert!(blowup_factor.is_power_of_two());
    let lde_size = n * blowup_factor;
    assert_u32_domain(
        lde_size,
        "coset_lde_batch_base_into_with_merkle_tree lde_size",
    );
    for o in outputs.iter() {
        assert_eq!(o.len(), lde_size);
    }
    assert!(
        rows_per_leaf == 1 || rows_per_leaf == 2,
        "rows_per_leaf must be 1 or 2"
    );
    assert_eq!(lde_size % rows_per_leaf, 0);
    let num_leaves = lde_size / rows_per_leaf;
    let nodes_dev_bytes = commit.total_nodes_bytes(num_leaves);
    assert_eq!(nodes_out.len(), nodes_dev_bytes);
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(m * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(m * lde_size) };

    // Pack columns into the pinned buffer. Runs under the pinned-staging
    // lock, where rayon can deadlock. See `Backend::pinned_staging`.
    {
        for (c, col) in columns.iter().enumerate() {
            pinned[c * n..c * n + n].copy_from_slice(col);
        }
    }

    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    {
        for c in 0..m {
            let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
            stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
        }
    }
    // The uploads above are truly asynchronous (pinned source); the staging
    // slot stays locked until this event fires (waited before the drain).
    staging.record_event(&stream)?;

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let m_u32 = m as u32;

    // iNTT
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        col_stride_u64,
        m_u32,
    )?;
    launch_pointwise_mul_batched(
        stream.as_ref(),
        be,
        &mut buf,
        &weights_dev,
        n_u64,
        col_stride_u64,
        m_u32,
    )?;
    // forward NTT at LDE size
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        m_u32,
    )?;

    // Allocate the device output buffer. In `LeavesOnly` mode this is just
    // `num_leaves * 32` bytes (the leaves themselves); in `FullTree` mode it's
    // `(2*num_leaves - 1) * 32` bytes (leaves in the tail + inner nodes filled
    // below). `alloc` (not `alloc_zeros`) is safe because every byte is
    // written before any reader sees it: the keccak kernel fills the
    // leaves slab, the inner-tree pass (when present) fills the head.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(nodes_dev_bytes) }?;
    let leaves_offset_bytes = commit.leaves_offset_bytes(num_leaves);
    {
        let mut leaves_view =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
        if rows_per_leaf == 2 {
            launch_keccak_base_row_pair(
                stream.as_ref(),
                &buf,
                col_stride_u64,
                m as u64,
                lde_u64,
                &mut leaves_view,
            )?;
        } else {
            launch_keccak_base(
                stream.as_ref(),
                &buf,
                col_stride_u64,
                m as u64,
                lde_u64,
                &mut leaves_view,
            )?;
        }
    }

    if commit == KeccakCommit::FullTree {
        crate::merkle::build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
    }

    // Release the staging slot before the drain: the uploads have landed once
    // the slot event fires (the kernels above are queued behind them).
    staging.sync_event()?;
    drop(staging);

    // D2H the LDE (async, via pinned staging, enqueued without blocking) and
    // the tree/leaves nodes (via the separate pinned-hashes slot; that helper
    // waits internally, and its event is recorded after the LDE copy, so the
    // `wait_and_read` below is nearly instant).
    let lde_pending =
        crate::device::async_dtoh_via(&stream, staging_slot, &be.ctx, &buf, m * lde_size)?;
    d2h_bytes_via_pinned_hashes(&stream, be, &nodes_dev, nodes_out)?;

    // Copy pinned into caller outputs. Runs under the pinned-staging lock
    // (held by `lde_pending`), where rayon can deadlock. See
    // `Backend::pinned_staging`.
    {
        lde_pending.wait_and_read(|bytes| {
            // SAFETY: the pinned slab is u64-aligned by construction and the
            // copy deposited exactly `m * lde_size` u64s.
            let pinned =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, m * lde_size) };
            for (c, dst) in outputs.iter_mut().enumerate() {
                dst.copy_from_slice(&pinned[c * lde_size..c * lde_size + lde_size]);
            }
        })?;
    }

    if keep_device_buf {
        Ok(Some(GpuLdeBase {
            buf: Arc::new(buf),
            m,
            lde_size,
            tree: None,
            trace_dev: None,
            trace_rows: 0,
            // The pending wait above drained the stream past the last write
            // to `buf`, so the handle is complete at return.
            ready: None,
        }))
    } else {
        drop(buf);
        Ok(None)
    }
}

/// Batched ext3 polynomial → coset evaluation.
///
/// Input: M ext3 columns of `n` coefficients each (interleaved, 3n u64).
/// Output: M ext3 columns of `n * blowup_factor` evaluations each at the
/// offset-coset.
///
/// Skips the iFFT stage of [`coset_lde_batch_ext3_into`] (input is
/// coefficients, not evaluations). Weights encode the coset shift:
/// `weights[k] = offset^k` (NO 1/N because iFFT normalisation doesn't apply).
pub fn evaluate_poly_coset_batch_ext3_into(
    coefs: &[&[u64]],
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
) -> Result<()> {
    evaluate_poly_coset_batch_ext3_into_inner(
        coefs,
        n,
        blowup_factor,
        weights,
        outputs,
        None,
        false,
    )
    .map(|_| ())
}

/// Same as [`evaluate_poly_coset_batch_ext3_into`] but retains the de-
/// interleaved LDE device buffer as a `GpuLdeExt3` handle so callers can
/// reuse the LDE without a re-H2D.
pub fn evaluate_poly_coset_batch_ext3_into_keep(
    coefs: &[&[u64]],
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
) -> Result<GpuLdeExt3> {
    let opt = evaluate_poly_coset_batch_ext3_into_inner(
        coefs,
        n,
        blowup_factor,
        weights,
        outputs,
        None,
        true,
    )?;
    Ok(opt.expect("keep_device_buf=true must return Some"))
}

fn evaluate_poly_coset_batch_ext3_into_inner(
    coefs: &[&[u64]],
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    merkle_nodes_out: Option<&mut [u8]>,
    keep_device_buf: bool,
) -> Result<Option<GpuLdeExt3>> {
    if coefs.is_empty() {
        assert_eq!(outputs.len(), 0);
        return Ok(None);
    }
    let m = coefs.len();
    assert_eq!(outputs.len(), m);
    // Empty domain must short-circuit before the power-of-two assert
    // (is_power_of_two returns false for 0).
    if n == 0 {
        return Ok(None);
    }
    assert!(n.is_power_of_two());
    assert_eq!(weights.len(), n);
    assert!(blowup_factor.is_power_of_two());
    for c in coefs.iter() {
        assert_eq!(c.len(), 3 * n);
    }
    let lde_size = n * blowup_factor;
    for o in outputs.iter() {
        assert_eq!(o.len(), 3 * lde_size);
    }
    assert_u32_domain(lde_size, "evaluate_poly_coset_batch_ext3_into lde_size");
    if merkle_nodes_out.is_some() {
        assert!(lde_size >= 2);
    }
    let log_lde = lde_size.trailing_zeros() as u64;

    let mb = 3 * m;
    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(mb * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(mb * lde_size) };

    pack_ext3_to_pinned_slabs(coefs, pinned, n);

    let mut buf = stream.alloc_zeros::<u64>(mb * lde_size)?;
    {
        for s in 0..mb {
            let mut dst = buf.slice_mut(s * lde_size..s * lde_size + n);
            stream.memcpy_htod(&pinned[s * n..s * n + n], &mut dst)?;
        }
    }
    // The uploads above are truly asynchronous (pinned source); the staging
    // slot stays locked until this event fires (waited before the drain).
    staging.record_event(&stream)?;

    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let mb_u32 = mb as u32;

    // Apply coset scaling: x[k] *= weights[k] for k in 0..n (no iFFT first).
    launch_pointwise_mul_batched(
        stream.as_ref(),
        be,
        &mut buf,
        &weights_dev,
        n_u64,
        col_stride_u64,
        mb_u32,
    )?;

    // Bit-reverse full lde_size slab, then forward DIT NTT.
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        lde_u64,
        log_lde,
        col_stride_u64,
        mb_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        mb_u32,
    )?;

    // Optional R2-style row-pair Merkle tree build on the LDE buffer, queued
    // ahead of the drains below.
    let nodes = if let Some(nodes_out) = merkle_nodes_out {
        let num_leaves = lde_size / 2;
        let tight_total_nodes = 2 * num_leaves - 1;
        assert_eq!(nodes_out.len(), tight_total_nodes * 32);
        let mut nodes_dev = unsafe { stream.alloc::<u8>(tight_total_nodes * 32) }?;
        let leaves_offset_bytes = (num_leaves - 1) * 32;
        {
            let mut leaves_view =
                nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
            let log_num_rows = log_lde;
            let num_parts_u64 = m as u64;
            let cfg = keccak_launch_cfg(num_leaves as u64);
            unsafe {
                stream
                    .launch_builder(&be.keccak_comp_poly_leaves_ext3)
                    .arg(&buf)
                    .arg(&col_stride_u64)
                    .arg(&num_parts_u64)
                    .arg(&lde_u64)
                    .arg(&log_num_rows)
                    .arg(&mut leaves_view)
                    .launch(cfg)?;
            }
        }
        crate::merkle::build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
        Some((nodes_dev, nodes_out))
    } else {
        None
    };

    // Release the staging slot before the drain: the uploads have landed once
    // the slot event fires (the kernels above are queued behind them).
    staging.sync_event()?;
    drop(staging);

    // LDE drain enqueued without blocking. When a tree was built, its nodes
    // drain via the separate pinned-hashes slot; that helper waits internally,
    // and its event is recorded after the LDE copy, so the `wait_and_read`
    // below is nearly instant.
    let lde_pending =
        crate::device::async_dtoh_via(&stream, staging_slot, &be.ctx, &buf, mb * lde_size)?;
    if let Some((nodes_dev, nodes_out)) = nodes {
        d2h_bytes_via_pinned_hashes(&stream, be, &nodes_dev, nodes_out)?;
    }

    {
        lde_pending.wait_and_read(|bytes| {
            // SAFETY: the pinned slab is u64-aligned by construction and the
            // copy deposited exactly `mb * lde_size` u64s.
            let pinned =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, mb * lde_size) };
            unpack_pinned_slabs_to_ext3(pinned, outputs, lde_size);
        })?;
    }
    if keep_device_buf {
        Ok(Some(GpuLdeExt3 {
            buf: std::sync::Arc::new(buf),
            m,
            lde_size,
            tree: None,
            // The pending wait above drained the stream past the last write
            // to `buf`, so the handle is complete at return.
            ready: None,
        }))
    } else {
        drop(buf);
        Ok(None)
    }
}

/// Fused variant of [`evaluate_poly_coset_batch_ext3_into`]: in addition to
/// the LDE output, builds the R2 composition-polynomial Merkle tree on device
/// (row-pair Keccak leaves at bit-reversed indices + pair-hash inner tree).
///
/// Row-pair commit: each leaf hashes 2 bit-reversed rows, so the tree has
/// `lde_size / 2` leaves and `merkle_nodes_out` must have byte length
/// `(lde_size - 1) * 32`. Requires `lde_size >= 2`.
pub fn evaluate_poly_coset_batch_ext3_into_with_merkle_tree(
    coefs: &[&[u64]],
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    merkle_nodes_out: &mut [u8],
) -> Result<()> {
    evaluate_poly_coset_batch_ext3_into_inner(
        coefs,
        n,
        blowup_factor,
        weights,
        outputs,
        Some(merkle_nodes_out),
        false,
    )
    .map(|_| ())
}
/// Batched coset LDE for Goldilocks **cubic extension** columns.
///
/// A degree-3 extension element is `(a, b, c)` in memory (three contiguous
/// u64s). The NTT butterfly multiplies `v = (a, b, c)` by a base-field
/// twiddle `t`: `t * v = (t*a, t*b, t*c)`. Addition is componentwise. So an
/// NTT over M ext3 columns is algebraically equivalent to **3M parallel
/// base-field NTTs** sharing the same twiddles and coset weights. We
/// exploit this to reuse the base-field kernels with no modification:
///
/// 1. Host pack de-interleaves each ext3 column into 3 consecutive
///    base-field slabs inside the pinned staging buffer (slab 0 has all the
///    a-components, slab 1 all the b's, slab 2 all the c's — 3M base slabs
///    in total).
/// 2. Existing `bit_reverse_permute_batched` / `ntt_dit_*_batched` /
///    `pointwise_mul_batched` run over those 3M base slabs on device.
/// 3. D2H, then re-interleave 3 slabs per output ext3 column.
///
/// Input/output layout: each slice is 3*n or 3*n*blowup u64s, packed as
/// `[a0, b0, c0, a1, b1, c1, ...]` — the natural `[FieldElement<Ext3>]`
/// memory representation.
pub fn coset_lde_batch_ext3_into(
    columns: &[&[u64]],
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
) -> Result<()> {
    if columns.is_empty() {
        return Ok(());
    }
    let m = columns.len();
    assert_eq!(outputs.len(), m, "outputs must match columns count");
    // Empty domain must short-circuit before the power-of-two assert
    // (is_power_of_two returns false for 0).
    if n == 0 {
        return Ok(());
    }
    assert!(n.is_power_of_two(), "n must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match n");
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup must be power of two"
    );
    for c in columns.iter() {
        assert_eq!(c.len(), 3 * n, "each ext3 column must be 3*n u64s");
    }
    let lde_size = n * blowup_factor;
    for o in outputs.iter() {
        assert_eq!(o.len(), 3 * lde_size, "each output must be 3*lde_size u64s");
    }
    assert_u32_domain(lde_size, "coset_lde_batch_ext3_into lde_size");
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    // 3 base slabs per ext3 column; slab index `c*3 + k` holds component `k`.
    let mb = 3 * m;

    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(mb * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(mb * lde_size) };

    pack_ext3_to_pinned_slabs(columns, pinned, n);

    // Allocate + zero-pad device buffer holding 3M slabs of `lde_size`.
    let mut buf = stream.alloc_zeros::<u64>(mb * lde_size)?;
    // H2D: slab by slab into the first N slots of each `lde_size`-slab.
    {
        for s in 0..mb {
            let mut dst = buf.slice_mut(s * lde_size..s * lde_size + n);
            stream.memcpy_htod(&pinned[s * n..s * n + n], &mut dst)?;
        }
    }
    // The uploads above are truly asynchronous (pinned source); the staging
    // slot stays locked until this event fires (waited before the drain).
    staging.record_event(&stream)?;

    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let mb_u32 = mb as u32;

    // === Butterflies: identical to the base-field batched path, but with
    // grid.y = 3M instead of M. ===
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        n_u64,
        log_n,
        col_stride_u64,
        mb_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        col_stride_u64,
        mb_u32,
    )?;
    launch_pointwise_mul_batched(
        stream.as_ref(),
        be,
        &mut buf,
        &weights_dev,
        n_u64,
        col_stride_u64,
        mb_u32,
    )?;
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        lde_u64,
        log_lde,
        col_stride_u64,
        mb_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        mb_u32,
    )?;

    // Release the staging slot before the drain: the uploads have landed once
    // the slot event fires (the kernels above are queued behind them).
    staging.sync_event()?;
    drop(staging);

    // Big D2H enqueued without blocking; the host blocks once, in
    // `wait_and_read` below.
    let pending =
        crate::device::async_dtoh_via(&stream, staging_slot, &be.ctx, &buf, mb * lde_size)?;

    // Unpack: for each output column, re-interleave 3 slabs back into the
    // ext3-per-element layout. Runs under the pinned-staging lock (held by
    // `pending`), where rayon can deadlock. See `Backend::pinned_staging`.
    {
        pending.wait_and_read(|bytes| {
            // SAFETY: the pinned slab is u64-aligned by construction and the
            // copy deposited exactly `mb * lde_size` u64s.
            let pinned =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, mb * lde_size) };
            unpack_pinned_slabs_to_ext3(pinned, outputs, lde_size);
        })?;
    }
    Ok(())
}

/// Batched ext3 coset LDE over columns ALREADY resident on device in slab
/// layout (`3m` slabs of `lde_size` u64, first `n` of each filled, rest
/// zero-padded), e.g. from the on-device degree-2 decomposition. Runs the
/// same butterfly pipeline as [`coset_lde_batch_ext3_into`], drains the
/// evaluations to `outputs` (interleaved ext3, `3*lde_size` u64 each), and
/// keeps the device buffer as a [`GpuLdeExt3`] handle (synchronized by the
/// drain, so `ready: None`).
pub fn coset_lde_batch_ext3_slabs_keep(
    stream: &Arc<CudaStream>,
    mut buf: CudaSlice<u64>,
    m: usize,
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
) -> Result<GpuLdeExt3> {
    assert!(m > 0 && n.is_power_of_two(), "slab LDE shape");
    assert_eq!(weights.len(), n, "weights length must match n");
    assert!(blowup_factor.is_power_of_two(), "blowup must be power of two");
    let lde_size = n * blowup_factor;
    let mb = 3 * m;
    assert_eq!(buf.len(), mb * lde_size, "slab buffer shape");
    assert_eq!(outputs.len(), m, "outputs must match column count");
    for o in outputs.iter() {
        assert_eq!(o.len(), 3 * lde_size, "each output must be 3*lde_size u64s");
    }
    assert_u32_domain(lde_size, "coset_lde_batch_ext3_slabs_keep lde_size");
    let log_n = n.trailing_zeros() as u64;
    let log_lde = lde_size.trailing_zeros() as u64;

    let be = backend()?;
    let inv_tw = be.inv_twiddles_for(log_n)?;
    let fwd_tw = be.fwd_twiddles_for(log_lde)?;
    let weights_dev = stream.clone_htod(weights)?;

    let n_u64 = n as u64;
    let lde_u64 = lde_size as u64;
    let col_stride_u64 = lde_size as u64;
    let mb_u32 = mb as u32;

    launch_bit_reverse_batched(stream.as_ref(), be, &mut buf, n_u64, log_n, col_stride_u64, mb_u32)?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        inv_tw.as_ref(),
        n_u64,
        log_n,
        col_stride_u64,
        mb_u32,
    )?;
    launch_pointwise_mul_batched(
        stream.as_ref(),
        be,
        &mut buf,
        &weights_dev,
        n_u64,
        col_stride_u64,
        mb_u32,
    )?;
    launch_bit_reverse_batched(
        stream.as_ref(),
        be,
        &mut buf,
        lde_u64,
        log_lde,
        col_stride_u64,
        mb_u32,
    )?;
    run_batched_ntt_body(
        stream.as_ref(),
        &mut buf,
        fwd_tw.as_ref(),
        lde_u64,
        log_lde,
        col_stride_u64,
        mb_u32,
    )?;

    let pending =
        crate::device::async_dtoh_via(stream, be.pinned_staging(), &be.ctx, &buf, mb * lde_size)?;
    pending.wait_and_read(|bytes| {
        // SAFETY: the pinned slab is u64-aligned by construction and the copy
        // deposited exactly `mb * lde_size` u64s.
        let pinned =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, mb * lde_size) };
        unpack_pinned_slabs_to_ext3(pinned, outputs, lde_size);
    })?;

    Ok(GpuLdeExt3 {
        buf: Arc::new(buf),
        m,
        lde_size,
        tree: None,
        ready: None,
    })
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
    let be = backend()?;
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
