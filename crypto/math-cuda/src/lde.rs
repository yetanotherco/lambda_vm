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

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys;
use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtrMut, LaunchConfig, PushKernelArg,
};

use crate::DeviceHash;
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
enum TreeCommit {
    /// Only the keccak-256 leaves; no inner-tree build. Caller receives
    /// `num_leaves * 32` bytes.
    LeavesOnly,
    /// Full Merkle tree: leaves at the tail + inner nodes built on-device.
    /// Caller receives `(2*num_leaves - 1) * 32` bytes.
    FullTree,
}

impl TreeCommit {
    fn total_nodes_bytes(self, num_leaves: usize) -> usize {
        match self {
            TreeCommit::LeavesOnly => num_leaves * 32,
            TreeCommit::FullTree => (2 * num_leaves - 1) * 32,
        }
    }

    fn leaves_offset_bytes(self, num_leaves: usize) -> usize {
        match self {
            TreeCommit::LeavesOnly => 0,
            TreeCommit::FullTree => (num_leaves - 1) * 32,
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
    // Levels 0..8 fused in shmem (one DRAM pass instead of eight); the
    // remaining high-stride levels keep one kernel per level.
    let mut first_level = 0u64;
    if n >= 256 {
        let t: u32 = 8.min(m as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: ((m as u32).div_ceil(t), ((n / 256) as u32).min(65535), 1),
            block_dim: (t, 128, 1),
            shared_mem_bytes: 256 * (t + 1) * 8,
        };
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_8_levels_row_major)
                .arg(&mut *buf)
                .arg(tw)
                .arg(&n)
                .arg(&log_n)
                .arg(&m)
                .launch(cfg)?;
        }
        first_level = 8.min(log_n);
    }

    let col_tile: u32 = 32.min(m as u32);
    let row_tile: u32 = (256 / col_tile).max(1);
    for level in first_level..log_n {
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

/// One `matrix_transpose_strided` launch: `dst[c * out_stride + r] =
/// src[r * cols + c]` for `r < rows`, `c < cols`. `src` holds `rows * cols`
/// contiguous row-major elements; `dst` must reach `(cols - 1) * out_stride +
/// rows`. No synchronize — callers on the same stream are ordered; other
/// streams must synchronize themselves.
fn launch_transpose_tiles(
    stream: &CudaStream,
    be: &Backend,
    src: &CudaView<'_, u64>,
    dst: &mut CudaViewMut<'_, u64>,
    rows: usize,
    cols: usize,
    out_stride: u64,
) -> Result<()> {
    debug_assert!(rows >= 1 && cols >= 1, "empty transpose");
    debug_assert!(src.len() >= rows * cols, "transpose source too short");
    debug_assert!(
        dst.len() >= (cols - 1) * out_stride as usize + rows,
        "transpose destination too short"
    );
    let cfg = LaunchConfig {
        grid_dim: (
            (cols as u32).div_ceil(32),
            (rows as u32).div_ceil(32).min(65535),
            1,
        ),
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.matrix_transpose_strided)
            .arg(src)
            .arg(dst)
            .arg(&(rows as u32))
            .arg(&(cols as u32))
            .arg(&out_stride)
            .launch(cfg)?;
    }
    Ok(())
}

/// Transpose the first `rows` row-major rows of `src` (`cols` wide) into a
/// NEW column-major buffer with column stride `out_stride` (the trace-domain
/// snapshot the LogUp fingerprint kernel reads). The LDE itself is never
/// transposed this way — see [`transpose_lde_in_place`], which keeps one
/// LDE-sized buffer live instead of two.
fn launch_row_to_col_major(
    stream: &Arc<CudaStream>,
    be: &Backend,
    src: &CudaSlice<u64>,
    rows: usize,
    cols: usize,
    out_stride: u64,
) -> Result<CudaSlice<u64>> {
    let mut dst = stream.alloc_zeros::<u64>(out_stride as usize * cols)?;
    launch_transpose_tiles(
        stream,
        be,
        &src.slice(0..rows * cols),
        &mut dst.slice_mut(..),
        rows,
        cols,
        out_stride,
    )?;
    Ok(dst)
}

// ── In-place row-major → column-major transpose of the LDE ──────────────────
//
// The fused commit computes the LDE row-major (one H2D, row-major NTT and leaf
// kernels) but every downstream kernel reads it column-major. Transposing into
// a fresh buffer held TWO LDE-sized allocations live at once — the peak of the
// whole per-table commit, and the term that pushed 2^21-row tables past 32 GiB.
// Here the transpose happens inside the one allocation, in two passes over the
// same `matrix_transpose_strided` kernel plus device-to-device copies, so the
// bytes that land are exactly the ones the out-of-place kernel used to write:
//
//  1. Block pass. The `rows × cols` matrix is `blocks` row blocks of
//     `rows_per_block` rows. Each block is transposed on its own into `cols`
//     runs of `rows_per_block` consecutive rows of one column. Blocks ping-pong
//     through one spare block of scratch: block 0 lands in the scratch, block
//     `b` lands where block `b - 1` was, and the scratch finally lands in the
//     last slot — so slot `s` holds block `(s + 1) mod blocks`.
//  2. Run pass. Column-major wants the runs ordered `(column, block)`; the
//     block pass left them ordered `(slot, column)`. That is a permutation of
//     whole runs, followed cycle by cycle in place with the scratch runs as the
//     parking space, and issued as batched device copies.
//
// Scratch is one block plus one run — capped at
// `INPLACE_TRANSPOSE_SCRATCH_BYTES` — instead of a second LDE.

/// Cap on the in-place transpose's device scratch: one transposed row block
/// (`cols` runs) plus one parking run for the cycle walk.
const INPLACE_TRANSPOSE_SCRATCH_BYTES: usize = 256 << 20;

/// Row-block geometry of the in-place transpose: `rows == blocks *
/// rows_per_block`, both powers of two, `blocks >= 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TransposeGeometry {
    rows_per_block: usize,
    blocks: usize,
}

fn transpose_geometry(rows: usize, cols: usize) -> TransposeGeometry {
    debug_assert!(rows >= 2 && rows.is_power_of_two(), "rows: {rows}");
    debug_assert!(cols >= 1, "cols: {cols}");
    // Longest power-of-two run whose scratch (`cols + 1` runs) fits the cap.
    let run_cap = (INPLACE_TRANSPOSE_SCRATCH_BYTES / ((cols + 1) * 8)).max(1);
    let run_cap = 1usize << run_cap.ilog2();
    // At least eight blocks whenever the matrix has the rows for it, so the
    // small shapes the parity suites drive take the same multi-block
    // permutation as production instead of a trivial two-block one.
    let rows_per_block = run_cap.min((rows / 8).max(1));
    TransposeGeometry {
        rows_per_block,
        blocks: rows / rows_per_block,
    }
}

/// A run — `rows_per_block` consecutive rows of one column — by run index,
/// either inside the LDE buffer or inside the scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Run {
    Buf(usize),
    Scratch(usize),
}

/// Sink for the run copies the permutation issues. The pairs inside one
/// `batch` call are independent — destinations are distinct and no destination
/// aliases a source of the same batch — so a sink may execute them in any
/// order or concurrently. Successive batches are ordered.
trait RunCopier {
    fn batch(&mut self, copies: &[(Run, Run)]) -> Result<()>;
}

/// After the block pass, run position `p = slot * cols + c` holds column `c`
/// of block `(slot + 1) mod blocks`; column-major needs that run at
/// `c * blocks + block`. This is the inverse map: the position whose run must
/// end up at `q`.
fn run_source_of(q: usize, blocks: usize, cols: usize) -> usize {
    let (c, block) = (q / blocks, q % blocks);
    ((block + blocks - 1) % blocks) * cols + c
}

/// Move every run to its column-major position, in place, cycle by cycle.
/// A cycle `[p0, p1, ..]` (run `p_{i+1}` moves to `p_i`) is served in
/// segments of at most `cols` moves: one batch parks the segment's sources in
/// scratch runs `0..m`, the next lands them; run `p0` waits in scratch run
/// `cols` until the cycle closes. About `2 * blocks + 3 * cycles` batches.
fn permute_runs_to_col_major(
    blocks: usize,
    cols: usize,
    copier: &mut impl RunCopier,
) -> Result<()> {
    let total = blocks * cols;
    let parked = Run::Scratch(cols);
    let mut visited = vec![false; total];
    let mut cycle: Vec<usize> = Vec::new();
    let mut copies: Vec<(Run, Run)> = Vec::with_capacity(cols + 1);
    for p0 in 0..total {
        if visited[p0] {
            continue;
        }
        cycle.clear();
        let mut p = p0;
        loop {
            visited[p] = true;
            cycle.push(p);
            p = run_source_of(p, blocks, cols);
            if p == p0 {
                break;
            }
        }
        let k = cycle.len();
        if k == 1 {
            continue;
        }
        copier.batch(&[(parked, Run::Buf(p0))])?;
        let mut i = 0;
        while i < k - 1 {
            let m = cols.min(k - 1 - i);
            copies.clear();
            copies.extend((0..m).map(|j| (Run::Scratch(j), Run::Buf(cycle[i + 1 + j]))));
            copier.batch(&copies)?;
            copies.clear();
            copies.extend((0..m).map(|j| (Run::Buf(cycle[i + j]), Run::Scratch(j))));
            if i + m == k - 1 {
                copies.push((Run::Buf(cycle[k - 1]), parked));
            }
            copier.batch(&copies)?;
            i += m;
        }
    }
    Ok(())
}

/// `LAMBDA_VM_LDE_TRANSPOSE_UNBATCHED=1` issues the run pass as one
/// device-to-device copy per run instead of `cuMemcpyBatchAsync` batches.
/// Same bytes either way; a measurement and diagnostic knob only.
fn transpose_copies_batched() -> bool {
    static BATCHED: OnceLock<bool> = OnceLock::new();
    *BATCHED.get_or_init(|| std::env::var_os("LAMBDA_VM_LDE_TRANSPOSE_UNBATCHED").is_none())
}

/// Issues run copies on the device: a batch is one `cuMemcpyBatchAsync`
/// (source access in stream order, both sides device memory), or one
/// `cuMemcpyDtoDAsync` per run when unbatched or for a single copy. The
/// caller keeps the context bound to this thread and both pointers valid
/// while the copier lives.
struct DeviceRunCopier<'a> {
    stream: &'a CudaStream,
    device: sys::CUdevice,
    buf: sys::CUdeviceptr,
    scratch: sys::CUdeviceptr,
    run_bytes: usize,
    batched: bool,
    dsts: Vec<sys::CUdeviceptr>,
    srcs: Vec<sys::CUdeviceptr>,
    sizes: Vec<usize>,
}

impl DeviceRunCopier<'_> {
    fn addr(&self, run: Run) -> sys::CUdeviceptr {
        match run {
            Run::Buf(p) => self.buf + (p * self.run_bytes) as sys::CUdeviceptr,
            Run::Scratch(j) => self.scratch + (j * self.run_bytes) as sys::CUdeviceptr,
        }
    }
}

impl RunCopier for DeviceRunCopier<'_> {
    fn batch(&mut self, copies: &[(Run, Run)]) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        if !self.batched || copies.len() == 1 {
            for &(dst, src) in copies {
                // SAFETY: both addresses lie inside allocations the caller
                // keeps alive (`buf`, `scratch`), each run is `run_bytes`
                // long, and the copy is queued on the caller's stream.
                unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        self.addr(dst),
                        self.addr(src),
                        self.run_bytes,
                        self.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
            return Ok(());
        }
        self.dsts.clear();
        self.srcs.clear();
        self.sizes.clear();
        for &(dst, src) in copies {
            self.dsts.push(self.addr(dst));
            self.srcs.push(self.addr(src));
            self.sizes.push(self.run_bytes);
        }
        let location = sys::CUmemLocation {
            type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
            id: self.device,
        };
        let mut attrs = sys::CUmemcpyAttributes {
            srcAccessOrder: sys::CUmemcpySrcAccessOrder_enum::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            srcLocHint: location,
            dstLocHint: location,
            flags: 0,
        };
        let mut attrs_idxs = [0usize];
        let mut fail_idx = 0usize;
        // SAFETY: the three arrays are `copies.len()` long and outlive the
        // call; one attribute set covers the whole batch (`attrsIdxs[0] ==
        // 0`); every address is inside `buf` or `scratch`, which the caller
        // keeps alive; the batch is queued on the caller's stream, and the
        // pairs are independent (the `RunCopier` contract).
        unsafe {
            sys::cuMemcpyBatchAsync(
                self.dsts.as_mut_ptr(),
                self.srcs.as_mut_ptr(),
                self.sizes.as_mut_ptr(),
                copies.len(),
                &mut attrs,
                attrs_idxs.as_mut_ptr(),
                1,
                &mut fail_idx,
                self.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }
}

/// Transpose the row-major `rows × cols` LDE in `buf` to column-major
/// (column `c` at `c * rows`) IN PLACE and hand the same allocation back.
/// Byte for byte the result of the out-of-place kernel — the two passes only
/// move runs of values — with one block plus one run of scratch instead of a
/// second LDE. Everything is queued on `stream`; nothing synchronizes.
fn transpose_lde_in_place(
    stream: &Arc<CudaStream>,
    be: &Backend,
    mut buf: CudaSlice<u64>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<u64>> {
    assert_eq!(buf.len(), rows * cols, "in-place transpose shape");
    if rows < 2 || cols < 2 {
        // A single row or a single column reads the same in both layouts.
        return Ok(buf);
    }
    let TransposeGeometry {
        rows_per_block,
        blocks,
    } = transpose_geometry(rows, cols);
    let block_elems = rows_per_block * cols;
    // `alloc`, not `alloc_zeros`: every scratch element is written before it is
    // read — the block pass fills the spare block, and each parking batch
    // fills the runs the next batch drains.
    let mut scratch = unsafe { stream.alloc::<u64>((cols + 1) * rows_per_block) }?;

    // Block pass: block 0 → scratch, block b → slot b - 1, scratch → last slot.
    {
        let src = buf.slice(0..block_elems);
        let mut spare = scratch.slice_mut(0..block_elems);
        launch_transpose_tiles(
            stream,
            be,
            &src,
            &mut spare,
            rows_per_block,
            cols,
            rows_per_block as u64,
        )?;
    }
    for block in 1..blocks {
        let (mut lo, hi) = buf.split_at_mut(block * block_elems);
        let src = hi.slice(0..block_elems);
        let mut dst = lo.slice_mut((block - 1) * block_elems..block * block_elems);
        launch_transpose_tiles(
            stream,
            be,
            &src,
            &mut dst,
            rows_per_block,
            cols,
            rows_per_block as u64,
        )?;
    }
    stream.memcpy_dtod(
        &scratch.slice(0..block_elems),
        &mut buf.slice_mut((blocks - 1) * block_elems..blocks * block_elems),
    )?;

    // Run pass: raw driver copies, so the context must be current here.
    be.ctx.bind_to_thread()?;
    {
        let (buf_ptr, _buf_record) = buf.device_ptr_mut(stream);
        let (scratch_ptr, _scratch_record) = scratch.device_ptr_mut(stream);
        let mut copier = DeviceRunCopier {
            stream,
            device: be.ctx.cu_device(),
            buf: buf_ptr,
            scratch: scratch_ptr,
            run_bytes: rows_per_block * 8,
            batched: transpose_copies_batched(),
            dsts: Vec::with_capacity(cols + 1),
            srcs: Vec::with_capacity(cols + 1),
            sizes: Vec::with_capacity(cols + 1),
        };
        permute_runs_to_col_major(blocks, cols, &mut copier)?;
    }
    // `scratch` drops here: freed stream-ordered behind the copies that read it.
    Ok(buf)
}

#[cfg(test)]
mod inplace_transpose_tests {
    use super::*;

    #[test]
    fn geometry_is_power_of_two_blocks_under_the_scratch_cap() {
        for log_rows in 1..=27u32 {
            let rows = 1usize << log_rows;
            for cols in [1usize, 2, 3, 9, 37, 316, 436, 612, 2048, 65535] {
                let g = transpose_geometry(rows, cols);
                assert!(g.rows_per_block.is_power_of_two());
                assert!(g.blocks.is_power_of_two());
                assert_eq!(g.rows_per_block * g.blocks, rows, "{rows}x{cols}");
                assert!(g.blocks >= 2, "{rows}x{cols}: {g:?}");
                assert!(
                    g.rows_per_block == 1
                        || (cols + 1) * g.rows_per_block * 8 <= INPLACE_TRANSPOSE_SCRATCH_BYTES,
                    "{rows}x{cols}: {g:?} busts the scratch cap"
                );
                if rows >= 8 && (cols + 1) * (rows / 8) * 8 <= INPLACE_TRANSPOSE_SCRATCH_BYTES {
                    assert_eq!(g.blocks, 8, "{rows}x{cols}: {g:?}");
                }
            }
        }
        // The production shapes the brief sizes: LDE 2^22 × 316 and 2^22 × 436.
        assert_eq!(
            transpose_geometry(1 << 22, 316),
            TransposeGeometry {
                rows_per_block: 1 << 16,
                blocks: 64
            }
        );
        assert_eq!(
            transpose_geometry(1 << 22, 436),
            TransposeGeometry {
                rows_per_block: 1 << 16,
                blocks: 64
            }
        );
    }

    /// Host model of the buffer after the block pass: one label `(block, col)`
    /// per run. Checks the independence contract on every batch.
    struct ModelCopier {
        buf: Vec<Option<(usize, usize)>>,
        scratch: Vec<Option<(usize, usize)>>,
        batches: usize,
        copies: usize,
    }

    impl ModelCopier {
        fn get(&self, run: Run) -> Option<(usize, usize)> {
            match run {
                Run::Buf(p) => self.buf[p],
                Run::Scratch(j) => self.scratch[j],
            }
        }
        fn set(&mut self, run: Run, v: Option<(usize, usize)>) {
            match run {
                Run::Buf(p) => self.buf[p] = v,
                Run::Scratch(j) => self.scratch[j] = v,
            }
        }
    }

    impl RunCopier for ModelCopier {
        fn batch(&mut self, copies: &[(Run, Run)]) -> Result<()> {
            for (i, (dst, src)) in copies.iter().enumerate() {
                assert!(
                    copies[..i].iter().all(|(d, _)| d != dst),
                    "duplicate destination {dst:?} in a batch"
                );
                assert!(
                    copies.iter().all(|(_, s)| s != dst),
                    "destination {dst:?} aliases a source of the same batch"
                );
                assert!(self.get(*src).is_some(), "copy from unwritten run {src:?}");
            }
            // Independent pairs: read everything, then write everything.
            let values: Vec<_> = copies.iter().map(|(_, src)| self.get(*src)).collect();
            for ((dst, _), v) in copies.iter().zip(values) {
                self.set(*dst, v);
            }
            self.batches += 1;
            self.copies += copies.len();
            Ok(())
        }
    }

    #[test]
    fn run_pass_lands_every_run_at_its_column_major_position() {
        for (blocks, cols) in [
            (2usize, 2usize),
            (2, 3),
            (4, 1),
            (4, 3),
            (8, 2),
            (8, 7),
            (8, 9),
            (8, 316),
            (8, 436),
            (64, 9),
            (64, 316),
            (64, 436),
            (128, 316),
            (128, 612),
            (16, 2048),
        ] {
            let mut model = ModelCopier {
                buf: (0..blocks * cols)
                    .map(|p| Some(((p / cols + 1) % blocks, p % cols)))
                    .collect(),
                scratch: vec![None; cols + 1],
                batches: 0,
                copies: 0,
            };
            permute_runs_to_col_major(blocks, cols, &mut model).unwrap();
            for q in 0..blocks * cols {
                assert_eq!(
                    model.buf[q],
                    Some((q % blocks, q / blocks)),
                    "{blocks}x{cols}: run {q}"
                );
            }
            // Cycle census of the same permutation, independently walked.
            let total = blocks * cols;
            let mut seen = vec![false; total];
            let (mut cycles, mut fixed) = (0usize, 0usize);
            for p0 in 0..total {
                if seen[p0] {
                    continue;
                }
                let (mut p, mut len) = (p0, 0usize);
                loop {
                    seen[p] = true;
                    len += 1;
                    p = run_source_of(p, blocks, cols);
                    if p == p0 {
                        break;
                    }
                }
                if len == 1 {
                    fixed += 1;
                } else {
                    cycles += 1;
                }
            }
            // Every moving run is parked once and landed once — no more.
            assert_eq!(model.copies, 2 * (total - fixed), "{blocks}x{cols}");
            // One parking batch per cycle, then two batches per `cols` moves.
            assert!(
                model.batches <= 2 * blocks + 3 * cycles,
                "{blocks}x{cols}: {} batches for {cycles} cycles",
                model.batches
            );
        }
    }
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
    // Big host traces go through the pinned staging slot: the driver's
    // internal pageable staging is 2-3x slower and convoys across threads.
    const PINNED_H2D_MIN_U64: usize = 1 << 20;
    let mut buf = stream.alloc_zeros::<u64>(lde_size * total_cols)?;
    match input {
        InnerInput::Host(h) if h.len() >= PINNED_H2D_MIN_U64 => {
            let mut dst = buf.slice_mut(0..n * total_cols);
            crate::device::htod_via(stream, be.pinned_staging(), &be.ctx, h, &mut dst)?;
        }
        InnerInput::Host(h) => stream.memcpy_htod(h, &mut buf.slice_mut(0..n * total_cols))?,
        InnerInput::Dev(d) => stream.memcpy_dtod(d, &mut buf.slice_mut(0..n * total_cols))?,
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

/// Shared row-major LDE + leaf-hash + Merkle pipeline for the base and ext3
/// paths, committing with the kernel family `hash` selects.
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
/// Walk the inner Merkle levels with the kernel family `hash` selects.
fn build_inner_tree_levels_for(
    hash: DeviceHash,
    stream: &CudaStream,
    be: &crate::device::Backend,
    nodes_dev: &mut CudaSlice<u8>,
    leaves_len: usize,
) -> Result<()> {
    match hash {
        DeviceHash::Keccak256 => {
            crate::merkle::build_inner_tree_levels(stream, be, nodes_dev, leaves_len)
        }
        DeviceHash::Blake3 => {
            crate::blake3::build_inner_tree_levels(stream, be, nodes_dev, leaves_len)
        }
        DeviceHash::Rpo256 | DeviceHash::Rpx256 | DeviceHash::Poseidon => {
            unimplemented!("{hash:?} device commit not yet ported (inner tree levels)")
        }
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn coset_lde_row_major_inner(
    input: InnerInput,
    hash: DeviceHash,
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
    let nodes_bytes = TreeCommit::FullTree.total_nodes_bytes(num_leaves);
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

    // Leaf hashing + Merkle on-device, with the kernel family `hash` selects.
    // Each row-pair leaf reads two bit-reversed rows of `total_cols` consecutive
    // u64s (`lde_u64` is the bit-reverse modulus; the kernel emits
    // `lde_size / 2` leaves).
    let mut nodes_dev = unsafe { stream.alloc::<u8>(nodes_bytes) }?;
    let leaves_offset = TreeCommit::FullTree.leaves_offset_bytes(num_leaves);
    {
        let mut leaves_view = nodes_dev.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
        match hash {
            DeviceHash::Keccak256 => launch_keccak_base_row_major_row_pair(
                stream.as_ref(),
                be,
                &buf,
                cols_u64,
                lde_u64,
                log_lde,
                &mut leaves_view,
            )?,
            DeviceHash::Blake3 => crate::blake3::launch_leaves_base_row_major_row_pair(
                stream.as_ref(),
                be,
                &buf,
                cols_u64,
                lde_u64,
                log_lde,
                &mut leaves_view,
            )?,
            DeviceHash::Rpo256 | DeviceHash::Rpx256 | DeviceHash::Poseidon => {
                unimplemented!("{hash:?} device commit not yet ported (row-major row-pair leaves)")
            }
        }
    }
    build_inner_tree_levels_for(hash, stream.as_ref(), be, &mut nodes_dev, num_leaves)?;

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

    // Transpose row-major buf into column-major for the handle, in place —
    // queued behind the D2H above, so the host copy sees the row-major bytes.
    // Downstream kernels (DEEP, barycentric) expect buf[c * lde_size + r].
    let col_major_dev = transpose_lde_in_place(&stream, be, buf, lde_size, total_cols)?;
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

/// Row-major LDE + leaf hashing + Merkle, all on-device, keeping the Merkle
/// tree
/// resident on device (in the handle's `tree`). The host tree is not built, so
/// the whole tree copy to host is eliminated; query openings gather paths from
/// the device tree.
///
/// Input: `row_major` is a flat `n * m` slice in row-major order; when
/// `predev` carries the same data already on device (pre-uploaded off the
/// critical path), the expansion D2D-copies from it instead of a fresh H2D.
/// Returns the `GpuLdeBase` handle (column-major buf, plus the device tree)
/// and the row-major LDE Vec.
#[allow(clippy::too_many_arguments)]
pub fn coset_lde_row_major_with_merkle_tree_keep(
    row_major: &[u64],
    predev: Option<&CudaSlice<u64>>,
    hash: DeviceHash,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_host_lde: bool,
) -> Result<(GpuLdeBase, Vec<u64>)> {
    let input = match predev {
        Some(d) if d.len() == row_major.len() => InnerInput::Dev(d),
        _ => InnerInput::Host(row_major),
    };
    let (tree, col_major_dev, lde_out, trace_col_major, ready) = coset_lde_row_major_inner(
        input,
        hash,
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
/// The precomputed tree's complete node buffer is downloaded to host
/// (`(2*num_leaves - 1) * 32` bytes, inner nodes first, root at offset 0,
/// leaves at the tail — the exact `MerkleTree::from_precomputed_nodes`
/// layout) because it feeds the process-wide host tree cache; it is only
/// built when `build_precomputed` is true (the caller skips it on a cache
/// hit). The multiplicity tree stays resident in `handle.tree` — openings
/// gather its paths on device.
///
/// Returns `(precomputed_nodes, handle, row_major_lde)`. The handle also
/// carries the column-major LDE + trace snapshot for downstream GPU rounds.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn coset_lde_row_major_split_trees(
    row_major: &[u64],
    predev: Option<&CudaSlice<u64>>,
    hash: DeviceHash,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    split_col: usize,
    build_precomputed: bool,
    retain_host_lde: bool,
) -> Result<(Option<Vec<u8>>, GpuLdeBase, Vec<u64>)> {
    assert!(split_col > 0 && split_col < m, "split inside the row");
    assert!(n.is_power_of_two(), "n must be a power of two");
    assert_eq!(weights.len(), n, "weights length must match n");
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup must be power of two"
    );
    assert_eq!(row_major.len(), n * m, "row-major input shape");
    let lde_size = n * blowup_factor;
    assert_u32_domain(lde_size, "coset_lde_row_major_split lde_size");
    let num_leaves = lde_size / 2;
    let nodes_bytes = TreeCommit::FullTree.total_nodes_bytes(num_leaves);
    let leaves_offset = TreeCommit::FullTree.leaves_offset_bytes(num_leaves);
    let log_lde = lde_size.trailing_zeros() as u64;
    let lde_u64 = lde_size as u64;
    let cols_u64 = m as u64;

    let be = backend()?;
    let stream = be.next_stream();

    let input = match predev {
        Some(d) if d.len() == row_major.len() => InnerInput::Dev(d),
        _ => InnerInput::Host(row_major),
    };
    let (buf, trace_col_major) =
        expand_row_major_on_stream(&stream, be, input, n, m, blowup_factor, weights, true)?;

    // One subset tree per column range, built sequentially on the stream.
    let build_subset_tree_dev = |col_start: u64, col_end: u64| -> Result<CudaSlice<u8>> {
        let mut nodes_dev = unsafe { stream.alloc::<u8>(nodes_bytes) }?;
        {
            let mut leaves_view =
                nodes_dev.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
            match hash {
                DeviceHash::Keccak256 => launch_keccak_base_row_major_row_pair_range(
                    stream.as_ref(),
                    be,
                    &buf,
                    cols_u64,
                    col_start,
                    col_end,
                    lde_u64,
                    log_lde,
                    &mut leaves_view,
                )?,
                DeviceHash::Blake3 => crate::blake3::launch_leaves_base_row_major_row_pair_range(
                    stream.as_ref(),
                    be,
                    &buf,
                    cols_u64,
                    col_start,
                    col_end,
                    lde_u64,
                    log_lde,
                    &mut leaves_view,
                )?,
                DeviceHash::Rpo256 | DeviceHash::Rpx256 | DeviceHash::Poseidon => unimplemented!(
                    "{hash:?} device commit not yet ported (row-major row-pair leaves, column range)"
                ),
            }
        }
        build_inner_tree_levels_for(hash, stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
        Ok(nodes_dev)
    };

    // Precomputed subset tree: full nodes to host (feeds the process-wide
    // host tree cache keyed by root; built once per prove on cache miss).
    let precomputed_nodes = if build_precomputed {
        let nodes_dev = build_subset_tree_dev(0, split_col as u64)?;
        let mut nodes_host = vec![0u8; nodes_bytes];
        stream.memcpy_dtoh(&nodes_dev, &mut nodes_host)?;
        Some(nodes_host)
    } else {
        None
    };
    // Multiplicity subset tree: resident (per-epoch; the ~2x-leaves node
    // download and host rebuild it used to pay are dropped — R4 openings
    // gather paths on device).
    let mult_tree = {
        let nodes_dev = build_subset_tree_dev(split_col as u64, cols_u64)?;
        let mut root = [0u8; 32];
        stream.memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
        GpuMerkleTree {
            nodes: Arc::new(nodes_dev),
            leaves_len: num_leaves,
            root,
        }
    };

    // D2H the row-major LDE only when the caller keeps a host copy; under
    // device-only every downstream consumer reads the handle.
    let lde_pending = retain_host_lde
        .then(|| {
            crate::device::async_dtoh_via(&stream, be.pinned_staging(), &be.ctx, &buf, lde_size * m)
        })
        .transpose()?;

    // Column-major handle for downstream GPU rounds (DEEP, barycentric,
    // constraint composition): transposed in place, behind the D2H above.
    let col_major_dev = transpose_lde_in_place(&stream, be, buf, lde_size, m)?;
    let ready = be.take_event()?;
    ready.event().record(&stream)?;

    let lde_out = match lde_pending {
        Some(pending) => {
            let mut out = vec![0u64; lde_size * m];
            pending.wait_into_u64(&mut out)?;
            out
        }
        None => Vec::new(),
    };

    let handle = GpuLdeBase {
        buf: Arc::new(col_major_dev),
        m,
        lde_size,
        tree: Some(mult_tree),
        ready: Some(Arc::new(ready)),
        trace_dev: trace_col_major.map(Arc::new),
        trace_rows: n,
    };
    Ok((precomputed_nodes, handle, lde_out))
}

/// Row-major ext3 LDE + leaf hashing + Merkle, all on-device.
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
    hash: DeviceHash,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_host_lde: bool,
) -> Result<(GpuLdeExt3, Vec<u64>)> {
    let (tree, col_major_dev, lde_out, _, ready) = coset_lde_row_major_inner(
        InnerInput::Host(row_major),
        hash,
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
    hash: DeviceHash,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[u64],
    retain_host_lde: bool,
) -> Result<(GpuLdeExt3, Vec<u64>)> {
    let (tree, col_major_dev, lde_out, _, ready) = coset_lde_row_major_inner(
        InnerInput::Dev(input_dev),
        hash,
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
    for (c, col) in columns.iter().enumerate() {
        pinned[c * n..c * n + n].copy_from_slice(col);
    }

    // Column layout: `buf[c * lde_size + r]`. Zeroed so the [n, lde_size)
    // tail of each column is already the zero-pad the CPU path does.
    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    // Any `?` between the first upload below and `sync_event` would release
    // the slot with async H2D reads of the slab still in flight; this guard
    // (declared after `staging`, so it drops first) drains the stream on
    // those error paths.
    let mut drain_on_err = crate::device::DrainOnErr {
        stream: &stream,
        armed: true,
    };
    // One memcpy per column from the pinned buffer into the strided slots.
    // The pinned source hits PCIe line-rate.
    for c in 0..m {
        let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
        stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
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
    drain_on_err.armed = false;
    drop(staging);

    // Single big D2H into the reusable pinned staging buffer — pinned, one
    // call to the driver, saturates PCIe. Enqueued without blocking; the host
    // blocks once, in `wait_and_read` below.
    let pending =
        crate::device::async_dtoh_via(&stream, staging_slot, &be.ctx, &buf, m * lde_size)?;

    // Split pinned into per-column Vec<u64>s. Runs under the pinned-staging
    // lock (held by `pending`), where rayon can deadlock. See
    // `Backend::pinned_staging`.
    let out: Vec<Vec<u64>> = pending.wait_and_read(|bytes| {
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
    })?;
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

    for (c, col) in columns.iter().enumerate() {
        pinned[c * n..c * n + n].copy_from_slice(col);
    }

    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    for c in 0..m {
        let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
        stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
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
    pending.wait_and_read(|bytes| {
        // SAFETY: the pinned slab is u64-aligned by construction and the
        // copy deposited exactly `m * lde_size` u64s.
        let pinned =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, m * lde_size) };
        for (c, dst) in outputs.iter_mut().enumerate() {
            dst.copy_from_slice(&pinned[c * lde_size..c * lde_size + lde_size]);
        }
    })?;
    Ok(())
}

/// Fused LDE + row-pair leaf hashing under the family `hash` selects.
/// Caller receives
/// `(lde_size / 2) * 32` bytes of leaf hashes in `hashed_leaves_out` (one
/// 32-byte digest per bit-reversed row pair, in natural leaf order, matching
/// `commit_bit_reversed(.., 2)` on the CPU side). Thin wrapper over
/// `coset_lde_batch_base_into_with_merkle_tree_inner` with `LeavesOnly` — no
/// inner-tree build, no device handle.
pub fn coset_lde_batch_base_into_with_leaf_hash(
    columns: &[&[u64]],
    hash: DeviceHash,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    hashed_leaves_out: &mut [u8],
) -> Result<()> {
    coset_lde_batch_base_into_with_merkle_tree_inner(
        columns,
        hash,
        blowup_factor,
        weights,
        outputs,
        hashed_leaves_out,
        TreeCommit::LeavesOnly,
        false,
        2,
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn coset_lde_batch_base_into_with_merkle_tree_inner(
    columns: &[&[u64]],
    hash: DeviceHash,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    nodes_out: &mut [u8],
    commit: TreeCommit,
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
    for (c, col) in columns.iter().enumerate() {
        pinned[c * n..c * n + n].copy_from_slice(col);
    }

    let mut buf = stream.alloc_zeros::<u64>(m * lde_size)?;
    for c in 0..m {
        let mut dst = buf.slice_mut(c * lde_size..c * lde_size + n);
        stream.memcpy_htod(&pinned[c * n..c * n + n], &mut dst)?;
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
        match (hash, rows_per_leaf == 2) {
            (DeviceHash::Keccak256, true) => launch_keccak_base_row_pair(
                stream.as_ref(),
                &buf,
                col_stride_u64,
                m as u64,
                lde_u64,
                &mut leaves_view,
            )?,
            (DeviceHash::Keccak256, false) => launch_keccak_base(
                stream.as_ref(),
                &buf,
                col_stride_u64,
                m as u64,
                lde_u64,
                &mut leaves_view,
            )?,
            (DeviceHash::Blake3, true) => crate::blake3::launch_leaves_base_row_pair(
                stream.as_ref(),
                &buf,
                col_stride_u64,
                m as u64,
                lde_u64,
                &mut leaves_view,
            )?,
            (DeviceHash::Blake3, false) => crate::blake3::launch_leaves_base(
                stream.as_ref(),
                &buf,
                col_stride_u64,
                m as u64,
                lde_u64,
                &mut leaves_view,
            )?,
            (DeviceHash::Rpo256 | DeviceHash::Rpx256 | DeviceHash::Poseidon, _) => {
                unimplemented!("{hash:?} device commit not yet ported (column-major base leaves)")
            }
        }
    }

    if commit == TreeCommit::FullTree {
        build_inner_tree_levels_for(hash, stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
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
    lde_pending.wait_and_read(|bytes| {
        // SAFETY: the pinned slab is u64-aligned by construction and the
        // copy deposited exactly `m * lde_size` u64s.
        let pinned =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, m * lde_size) };
        for (c, dst) in outputs.iter_mut().enumerate() {
            dst.copy_from_slice(&pinned[c * lde_size..c * lde_size + lde_size]);
        }
    })?;

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
        // No tree on this face (`merkle_nodes_out: None`): the hash dispatch
        // key is never read.
        DeviceHash::Keccak256,
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
        // No tree on this face (`merkle_nodes_out: None` below): the hash
        // dispatch key is never read.
        DeviceHash::Keccak256,
        n,
        blowup_factor,
        weights,
        outputs,
        None,
        true,
    )?;
    Ok(opt.expect("keep_device_buf=true must return Some"))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_poly_coset_batch_ext3_into_inner(
    coefs: &[&[u64]],
    hash: DeviceHash,
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
    for s in 0..mb {
        let mut dst = buf.slice_mut(s * lde_size..s * lde_size + n);
        stream.memcpy_htod(&pinned[s * n..s * n + n], &mut dst)?;
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
            match hash {
                DeviceHash::Keccak256 => {
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
                DeviceHash::Blake3 => crate::blake3::launch_comp_poly_leaves_ext3(
                    stream.as_ref(),
                    be,
                    &buf,
                    col_stride_u64,
                    num_parts_u64,
                    lde_u64,
                    log_num_rows,
                    &mut leaves_view,
                )?,
                DeviceHash::Rpo256 | DeviceHash::Rpx256 | DeviceHash::Poseidon => {
                    unimplemented!("{hash:?} device commit not yet ported (comp-poly ext3 leaves)")
                }
            }
        }
        build_inner_tree_levels_for(hash, stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
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

    lde_pending.wait_and_read(|bytes| {
        // SAFETY: the pinned slab is u64-aligned by construction and the
        // copy deposited exactly `mb * lde_size` u64s.
        let pinned =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, mb * lde_size) };
        unpack_pinned_slabs_to_ext3(pinned, outputs, lde_size);
    })?;
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
    hash: DeviceHash,
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: &mut [&mut [u64]],
    merkle_nodes_out: &mut [u8],
) -> Result<()> {
    evaluate_poly_coset_batch_ext3_into_inner(
        coefs,
        hash,
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
    for s in 0..mb {
        let mut dst = buf.slice_mut(s * lde_size..s * lde_size + n);
        stream.memcpy_htod(&pinned[s * n..s * n + n], &mut dst)?;
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
    pending.wait_and_read(|bytes| {
        // SAFETY: the pinned slab is u64-aligned by construction and the
        // copy deposited exactly `mb * lde_size` u64s.
        let pinned =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, mb * lde_size) };
        unpack_pinned_slabs_to_ext3(pinned, outputs, lde_size);
    })?;
    Ok(())
}

/// Batched ext3 coset LDE over columns ALREADY resident on device in slab
/// layout (`3m` slabs of `lde_size` u64, first `n` of each filled, rest
/// zero-padded), e.g. from the on-device degree-2 decomposition. Runs the
/// same butterfly pipeline as [`coset_lde_batch_ext3_into`] and keeps the
/// device buffer as a [`GpuLdeExt3`] handle. With `outputs = Some(..)` the
/// evaluations are also drained to host (interleaved ext3, `3*lde_size` u64
/// each; the drain synchronizes, so `ready: None`). With `None` nothing
/// leaves the device and the handle carries a `ready` event instead.
pub fn coset_lde_batch_ext3_slabs_keep(
    stream: &Arc<CudaStream>,
    mut buf: CudaSlice<u64>,
    m: usize,
    n: usize,
    blowup_factor: usize,
    weights: &[u64],
    outputs: Option<&mut [&mut [u64]]>,
) -> Result<GpuLdeExt3> {
    assert!(m > 0 && n.is_power_of_two(), "slab LDE shape");
    assert_eq!(weights.len(), n, "weights length must match n");
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup must be power of two"
    );
    let lde_size = n * blowup_factor;
    let mb = 3 * m;
    assert_eq!(buf.len(), mb * lde_size, "slab buffer shape");
    if let Some(outputs) = outputs.as_ref() {
        assert_eq!(outputs.len(), m, "outputs must match column count");
        for o in outputs.iter() {
            assert_eq!(o.len(), 3 * lde_size, "each output must be 3*lde_size u64s");
        }
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

    let ready = match outputs {
        Some(outputs) => {
            let pending = crate::device::async_dtoh_via(
                stream,
                be.pinned_staging(),
                &be.ctx,
                &buf,
                mb * lde_size,
            )?;
            pending.wait_and_read(|bytes| {
                // SAFETY: the pinned slab is u64-aligned by construction and the
                // copy deposited exactly `mb * lde_size` u64s.
                let pinned = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const u64, mb * lde_size)
                };
                unpack_pinned_slabs_to_ext3(pinned, outputs, lde_size);
            })?;
            None
        }
        None => {
            let ready = be.take_event()?;
            ready.event().record(stream)?;
            Some(Arc::new(ready))
        }
    };

    Ok(GpuLdeExt3 {
        buf: Arc::new(buf),
        m,
        lde_size,
        tree: None,
        ready,
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
