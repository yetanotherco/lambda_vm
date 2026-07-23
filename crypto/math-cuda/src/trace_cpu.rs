//! On-GPU CPU trace-table generation. Uploads packed `CpuOperation` fields and
//! fills the CPU table row-major on device (see `kernels/trace_cpu.cu`), leaving
//! the result resident so it feeds `coset_lde_row_major_with_merkle_tree_keep_dev`
//! with no host round-trip.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{Backend, backend};

/// CPU table width (`prover::tables::cpu::cols::NUM_COLUMNS`).
pub const CPU_NCOLS: usize = 38;
/// Packed input stride, u64 per op (must match `trace_cpu.cu`).
pub const CPU_OP_STRIDE: usize = 11;

/// Build one CPU trace-table chunk on device from packed ops.
///
/// `packed_ops` is `n * CPU_OP_STRIDE` u64s (see `trace_cpu.cu` for the field
/// layout). `num_rows` is the padded power-of-two row count and `last_ts` the
/// timestamp of the last real op (0 when `n == 0`). Returns the row-major device
/// buffer `[row*CPU_NCOLS + col]` (`num_rows * CPU_NCOLS` u64), ready for the
/// device-input LDE.
pub fn gpu_build_cpu_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
    last_ts: u64,
) -> Result<CudaSlice<u64>> {
    debug_assert_eq!(packed_ops.len(), n * CPU_OP_STRIDE);
    let be = backend()?;
    let stream = be.next_stream();

    // `clone_htod` rejects empty slices; a 1-element dummy is never read because
    // every row is a padding row when `n == 0`.
    let ops_dev = if packed_ops.is_empty() {
        stream.alloc_zeros::<u64>(CPU_OP_STRIDE)?
    } else {
        stream.clone_htod(packed_ops)?
    };

    // Zero-initialised: the kernel only writes non-zero cells.
    let mut out = stream.alloc_zeros::<u64>(num_rows * CPU_NCOLS)?;

    unsafe {
        stream
            .launch_builder(&be.trace_cpu_fill)
            .arg(&ops_dev)
            .arg(&(n as u64))
            .arg(&(num_rows as u64))
            .arg(&last_ts)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning wrapper over [`gpu_build_cpu_trace`] for byte-parity tests:
/// builds the row-major CPU table on device and copies it back on the same stream.
pub fn gpu_build_cpu_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
    last_ts: u64,
) -> Result<Vec<u64>> {
    debug_assert_eq!(packed_ops.len(), n * CPU_OP_STRIDE);
    let be = backend()?;
    let stream = be.next_stream();

    let ops_dev = if packed_ops.is_empty() {
        stream.alloc_zeros::<u64>(CPU_OP_STRIDE)?
    } else {
        stream.clone_htod(packed_ops)?
    };
    let mut out = stream.alloc_zeros::<u64>(num_rows * CPU_NCOLS)?;

    unsafe {
        stream
            .launch_builder(&be.trace_cpu_fill)
            .arg(&ops_dev)
            .arg(&(n as u64))
            .arg(&(num_rows as u64))
            .arg(&last_ts)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// MEMW_A table width (`prover::tables::memw_aligned::cols::NUM_COLUMNS`).
pub const MEMW_ALIGNED_NCOLS: usize = 29;
/// Packed MEMW_A input stride, u64 per op (must match `memw_aligned_fill` in `trace_cpu.cu`).
pub const MEMW_ALIGNED_STRIDE: usize = 12;

/// Build one MEMW_A (aligned memory) trace-table chunk on device from packed ops.
///
/// `packed_ops` is `n * MEMW_ALIGNED_STRIDE` u64s (see `trace_cpu.cu` for the field
/// layout). `num_rows` is the padded power-of-two row count. Returns the row-major
/// device buffer `[row*MEMW_ALIGNED_NCOLS + col]` (`num_rows * MEMW_ALIGNED_NCOLS`
/// u64), left resident so it feeds the device-input LDE with no full-column upload.
pub fn gpu_build_memw_aligned_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    debug_assert_eq!(packed_ops.len(), n * MEMW_ALIGNED_STRIDE);
    let be = backend()?;
    let stream = be.next_stream();

    let mut out = stream.alloc_zeros::<u64>(num_rows * MEMW_ALIGNED_NCOLS)?;

    // `clone_htod` rejects empty slices; a 1-op dummy is never read (all padding).
    let ops_dev = if packed_ops.is_empty() {
        stream.alloc_zeros::<u64>(MEMW_ALIGNED_STRIDE)?
    } else {
        stream.clone_htod(packed_ops)?
    };

    unsafe {
        stream
            .launch_builder(&be.memw_aligned_fill)
            .arg(&ops_dev)
            .arg(&(n as u64))
            .arg(&(num_rows as u64))
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok(out)
}

// -----------------------------------------------------------------------------
// LOAD / STORE (per-row-map memory tables — same shape as MEMW_A). A shared
// interleaved-fill launcher keeps the per-table code to a packing convention +
// column/stride constants.
// -----------------------------------------------------------------------------

/// LOAD table width / packed input stride (must match `load_fill` in `trace_cpu.cu`).
pub const LOAD_NCOLS: usize = 18;
pub const LOAD_STRIDE: usize = 7;
/// STORE table width / packed input stride (must match `store_fill`).
pub const STORE_NCOLS: usize = 16;
pub const STORE_STRIDE: usize = 4;
/// SHIFT table width / packed input stride (must match `shift_fill`).
pub const SHIFT_NCOLS: usize = 29;
pub const SHIFT_STRIDE: usize = 3;
/// LT table width / packed input stride (must match `lt_fill`). Input is already
/// host-deduplicated: one unique op + summed multiplicity per row.
pub const LT_NCOLS: usize = 17;
pub const LT_STRIDE: usize = 4;
/// EQ table width / packed input stride (must match `eq_fill`). Like LT, input is
/// host-deduplicated: one unique op + summed multiplicity per row.
pub const EQ_NCOLS: usize = 12;
pub const EQ_STRIDE: usize = 4;
/// BYTEWISE table width / packed input stride (must match `bytewise_fill`). Like
/// LT, input is host-deduplicated: one unique op + summed multiplicity per row.
pub const BYTEWISE_NCOLS: usize = 26;
pub const BYTEWISE_STRIDE: usize = 4;
/// MUL table width / packed input stride (must match `mul_fill`). Input is
/// host-deduplicated: one unique op + split (mu_lo, mu_hi) multiplicities per row.
pub const MUL_NCOLS: usize = 26;
pub const MUL_STRIDE: usize = 5;
/// DVRM table width / packed input stride (must match `dvrm_fill`). Input is
/// host-deduplicated: one unique op + split (mu_q, mu_r) multiplicities per row.
pub const DVRM_NCOLS: usize = 34;
pub const DVRM_STRIDE: usize = 5;
/// BRANCH table width / packed input stride (must match `branch_fill`). Input is
/// host-deduplicated: one unique op + summed multiplicity per row.
pub const BRANCH_NCOLS: usize = 14;
pub const BRANCH_STRIDE: usize = 5;
/// CPU32 table width / packed input stride (must match `cpu32_fill`). Per-row
/// (μ=1, no dedup): one op per row.
pub const CPU32_NCOLS: usize = 38;
pub const CPU32_STRIDE: usize = 8;
/// MEMW (general/unaligned) table width / packed input stride (must match
/// `memw_fill`). Per-row (no dedup): one walked op per row.
pub const MEMW_NCOLS: usize = 49;
pub const MEMW_STRIDE: usize = 19;

/// Shared interleaved fill on `stream`: upload `packed` (n × stride u64) and run
/// `kernel(ops, n, num_rows, out)` into a zeroed row-major buffer. Returns the
/// (unsynchronized) device buffer.
#[allow(clippy::too_many_arguments)]
fn build_interleaved_on(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
    ncols: usize,
    stride: usize,
) -> Result<CudaSlice<u64>> {
    debug_assert_eq!(packed_ops.len(), n * stride);
    let mut out = stream.alloc_zeros::<u64>(num_rows * ncols)?;
    let ops_dev = if packed_ops.is_empty() {
        stream.alloc_zeros::<u64>(stride)?
    } else {
        stream.clone_htod(packed_ops)?
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&ops_dev)
            .arg(&(n as u64))
            .arg(&(num_rows as u64))
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    Ok(out)
}

fn load_kernel(be: &Backend) -> &CudaFunction {
    &be.load_fill
}
fn store_kernel(be: &Backend) -> &CudaFunction {
    &be.store_fill
}
fn shift_kernel(be: &Backend) -> &CudaFunction {
    &be.shift_fill
}
fn lt_kernel(be: &Backend) -> &CudaFunction {
    &be.lt_fill
}
fn eq_kernel(be: &Backend) -> &CudaFunction {
    &be.eq_fill
}
fn bytewise_kernel(be: &Backend) -> &CudaFunction {
    &be.bytewise_fill
}
fn mul_kernel(be: &Backend) -> &CudaFunction {
    &be.mul_fill
}
fn dvrm_kernel(be: &Backend) -> &CudaFunction {
    &be.dvrm_fill
}
fn branch_kernel(be: &Backend) -> &CudaFunction {
    &be.branch_fill
}
fn cpu32_kernel(be: &Backend) -> &CudaFunction {
    &be.cpu32_fill
}
fn memw_kernel(be: &Backend) -> &CudaFunction {
    &be.memw_fill
}

/// Build one LT trace-table chunk on device from HOST-DEDUPLICATED ops (one unique
/// op + summed multiplicity per row; see `lt_fill` in `trace_cpu.cu`).
pub fn gpu_build_lt_trace(packed_ops: &[u64], n: usize, num_rows: usize) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        lt_kernel(be),
        packed_ops,
        n,
        num_rows,
        LT_NCOLS,
        LT_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning LT build for multiset-equality tests.
pub fn gpu_build_lt_trace_host(packed_ops: &[u64], n: usize, num_rows: usize) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        lt_kernel(be),
        packed_ops,
        n,
        num_rows,
        LT_NCOLS,
        LT_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one EQ trace-table chunk on device from HOST-DEDUPLICATED ops (one unique
/// op + summed multiplicity per row; see `eq_fill` in `trace_cpu.cu`).
pub fn gpu_build_eq_trace(packed_ops: &[u64], n: usize, num_rows: usize) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        eq_kernel(be),
        packed_ops,
        n,
        num_rows,
        EQ_NCOLS,
        EQ_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning EQ build for multiset-equality tests.
pub fn gpu_build_eq_trace_host(packed_ops: &[u64], n: usize, num_rows: usize) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        eq_kernel(be),
        packed_ops,
        n,
        num_rows,
        EQ_NCOLS,
        EQ_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one BYTEWISE trace-table chunk on device from HOST-DEDUPLICATED ops (one
/// unique op + summed multiplicity per row; see `bytewise_fill` in `trace_cpu.cu`).
pub fn gpu_build_bytewise_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        bytewise_kernel(be),
        packed_ops,
        n,
        num_rows,
        BYTEWISE_NCOLS,
        BYTEWISE_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning BYTEWISE build for multiset-equality tests.
pub fn gpu_build_bytewise_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        bytewise_kernel(be),
        packed_ops,
        n,
        num_rows,
        BYTEWISE_NCOLS,
        BYTEWISE_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one MUL trace-table chunk on device from HOST-DEDUPLICATED ops (one unique
/// op + split mu_lo/mu_hi multiplicities per row; see `mul_fill` in `trace_cpu.cu`).
pub fn gpu_build_mul_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        mul_kernel(be),
        packed_ops,
        n,
        num_rows,
        MUL_NCOLS,
        MUL_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning MUL build for multiset-equality tests.
pub fn gpu_build_mul_trace_host(packed_ops: &[u64], n: usize, num_rows: usize) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        mul_kernel(be),
        packed_ops,
        n,
        num_rows,
        MUL_NCOLS,
        MUL_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one DVRM trace-table chunk on device from HOST-DEDUPLICATED ops (one unique
/// op + split mu_q/mu_r multiplicities per row; see `dvrm_fill` in `trace_cpu.cu`).
pub fn gpu_build_dvrm_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        dvrm_kernel(be),
        packed_ops,
        n,
        num_rows,
        DVRM_NCOLS,
        DVRM_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning DVRM build for multiset-equality tests.
pub fn gpu_build_dvrm_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        dvrm_kernel(be),
        packed_ops,
        n,
        num_rows,
        DVRM_NCOLS,
        DVRM_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one BRANCH trace-table chunk on device from HOST-DEDUPLICATED ops (one
/// unique op + summed multiplicity per row; see `branch_fill` in `trace_cpu.cu`).
pub fn gpu_build_branch_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        branch_kernel(be),
        packed_ops,
        n,
        num_rows,
        BRANCH_NCOLS,
        BRANCH_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning BRANCH build for multiset-equality tests.
pub fn gpu_build_branch_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        branch_kernel(be),
        packed_ops,
        n,
        num_rows,
        BRANCH_NCOLS,
        BRANCH_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one CPU32 trace-table chunk on device (per-row `*W` fill; see `cpu32_fill`
/// in `trace_cpu.cu`).
pub fn gpu_build_cpu32_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        cpu32_kernel(be),
        packed_ops,
        n,
        num_rows,
        CPU32_NCOLS,
        CPU32_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning CPU32 build for byte-parity tests.
pub fn gpu_build_cpu32_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        cpu32_kernel(be),
        packed_ops,
        n,
        num_rows,
        CPU32_NCOLS,
        CPU32_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one MEMW (general/unaligned) trace-table chunk on device from walked ops
/// (see `memw_fill` in `trace_cpu.cu`).
pub fn gpu_build_memw_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        memw_kernel(be),
        packed_ops,
        n,
        num_rows,
        MEMW_NCOLS,
        MEMW_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning MEMW (general) build for byte-parity tests.
pub fn gpu_build_memw_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        memw_kernel(be),
        packed_ops,
        n,
        num_rows,
        MEMW_NCOLS,
        MEMW_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one SHIFT trace-table chunk on device (residency-ready row-major buffer).
/// The kernel recomputes the shift aux from the packed inputs (see `trace_cpu.cu`).
pub fn gpu_build_shift_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        shift_kernel(be),
        packed_ops,
        n,
        num_rows,
        SHIFT_NCOLS,
        SHIFT_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning SHIFT build for byte-parity tests.
pub fn gpu_build_shift_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        shift_kernel(be),
        packed_ops,
        n,
        num_rows,
        SHIFT_NCOLS,
        SHIFT_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one LOAD trace-table chunk on device (residency-ready row-major buffer).
pub fn gpu_build_load_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        load_kernel(be),
        packed_ops,
        n,
        num_rows,
        LOAD_NCOLS,
        LOAD_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning LOAD build for byte-parity tests.
pub fn gpu_build_load_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        load_kernel(be),
        packed_ops,
        n,
        num_rows,
        LOAD_NCOLS,
        LOAD_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build one STORE trace-table chunk on device (residency-ready row-major buffer).
pub fn gpu_build_store_trace(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        store_kernel(be),
        packed_ops,
        n,
        num_rows,
        STORE_NCOLS,
        STORE_STRIDE,
    )?;
    stream.synchronize()?;
    Ok(out)
}

/// Host-returning STORE build for byte-parity tests.
pub fn gpu_build_store_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let out = build_interleaved_on(
        &stream,
        store_kernel(be),
        packed_ops,
        n,
        num_rows,
        STORE_NCOLS,
        STORE_STRIDE,
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Host-returning wrapper over [`gpu_build_memw_aligned_trace`] for byte-parity
/// tests: builds the row-major MEMW_A buffer and copies it back on the same stream.
pub fn gpu_build_memw_aligned_trace_host(
    packed_ops: &[u64],
    n: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    debug_assert_eq!(packed_ops.len(), n * MEMW_ALIGNED_STRIDE);
    let be = backend()?;
    let stream = be.next_stream();

    let mut out = stream.alloc_zeros::<u64>(num_rows * MEMW_ALIGNED_NCOLS)?;
    let ops_dev = if packed_ops.is_empty() {
        stream.alloc_zeros::<u64>(MEMW_ALIGNED_STRIDE)?
    } else {
        stream.clone_htod(packed_ops)?
    };
    unsafe {
        stream
            .launch_builder(&be.memw_aligned_fill)
            .arg(&ops_dev)
            .arg(&(n as u64))
            .arg(&(num_rows as u64))
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

// -----------------------------------------------------------------------------
// MEMW_R (register fast-path) fill. Registers are pre-walked on the host (the
// sequential memory model recovers old_value/old_ts); this fills the 10 MEMW_R
// columns row-major on device from the walked rows, leaving the matrix resident
// for the device-input LDE (removes MEMW_R's full-column H2D). See the
// `memw_register_fill` kernel in `trace_cpu.cu`.
// -----------------------------------------------------------------------------

/// MEMW_R table width (`prover::tables::memw_register::cols::NUM_COLUMNS`).
pub const MEMW_REGISTER_NCOLS: usize = 10;

/// Build the MEMW_R trace table on device from already-walked rows (the
/// `old_value`/`old_ts` recovered by the host walk are uploaded here). Returns the
/// residency-ready row-major `[row*NCOLS+col]` buffer. `row_index` is the identity
/// (every input row is a real MEMW_R row).
#[allow(clippy::too_many_arguments)]
pub fn gpu_fill_memw_register(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    old_value: &[u64],
    old_ts: &[u64],
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let buf = fill_memw_register_on(
        &stream, reg_addr, ts, value, is_read, old_value, old_ts, num_rows,
    )?;
    stream.synchronize()?;
    Ok(buf)
}

/// A1 — build all resident PAGE tables on device from the initial image + the device final-memory
/// snapshot (with timestamps). One row-major `[page_size*ncols]` buffer per page (`page_bases`
/// ascending, one `page_fill_snapshot` launch each), left resident for the LDE. The sorted image +
/// snapshot upload ONCE. Bit-identical to `generate_page_trace_from_dense` (exclude_touched=false) —
/// replaces the ~285ms host PAGE column-fill.
#[allow(clippy::too_many_arguments)]
pub fn gpu_page_fill_snapshot(
    page_bases: &[u64],
    page_size: u64,
    img_addr: &[u64],
    img_val: &[u64],
    snap_addr: &[u64],
    snap_val: &[u64],
    snap_ts: &[u64],
    ncols: usize,
) -> Result<Vec<Vec<u64>>> {
    let be = backend()?;
    let stream = be.next_stream();
    let ia = stream.clone_htod(img_addr)?;
    let iv = stream.clone_htod(img_val)?;
    let sa = stream.clone_htod(snap_addr)?;
    let sv = stream.clone_htod(snap_val)?;
    let st = stream.clone_htod(snap_ts)?;
    let img_n = img_addr.len() as u64;
    let snap_n = snap_addr.len() as u64;
    let ncols_u32 = ncols as u32;
    // Fill each page's row-major table on device, then download to a host row-major `[row*ncols+col]`
    // matrix (the preprocessed commit + the R2 aux build both read the host matrix). Device fill
    // (~63ms) + download replaces the ~285ms host column fill; the LDE re-uploads in the commit phase.
    let mut bufs = Vec::with_capacity(page_bases.len());
    for &pb in page_bases {
        let mut buf = stream.alloc_zeros::<u64>(page_size as usize * ncols)?;
        unsafe {
            stream
                .launch_builder(&be.page_fill_snapshot)
                .arg(&pb)
                .arg(&page_size)
                .arg(&ia)
                .arg(&iv)
                .arg(&img_n)
                .arg(&sa)
                .arg(&sv)
                .arg(&st)
                .arg(&snap_n)
                .arg(&ncols_u32)
                .arg(&mut buf)
                .launch(LaunchConfig::for_num_elems(page_size as u32))?;
        }
        bufs.push(stream.clone_dtoh(&buf)?);
    }
    stream.synchronize()?;
    Ok(bufs)
}

/// Host-returning wrapper over [`gpu_fill_memw_register`] for byte-parity tests.
#[allow(clippy::too_many_arguments)]
pub fn gpu_fill_memw_register_host(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    old_value: &[u64],
    old_ts: &[u64],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let buf = fill_memw_register_on(
        &stream, reg_addr, ts, value, is_read, old_value, old_ts, num_rows,
    )?;
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

#[allow(clippy::too_many_arguments)]
fn fill_memw_register_on(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    old_value: &[u64],
    old_ts: &[u64],
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let n = reg_addr.len();
    debug_assert_eq!(ts.len(), n);
    debug_assert_eq!(value.len(), n);
    debug_assert_eq!(is_read.len(), n);
    debug_assert_eq!(old_value.len(), n);
    debug_assert_eq!(old_ts.len(), n);
    let be = backend()?;
    let mut buf = stream.alloc_zeros::<u64>(num_rows * MEMW_REGISTER_NCOLS)?;
    if n == 0 {
        return Ok(buf);
    }
    let keys_d = stream.clone_htod(reg_addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let is_read_d = stream.clone_htod(is_read)?;
    let old_value_d = stream.clone_htod(old_value)?;
    let old_ts_d = stream.clone_htod(old_ts)?;
    let row_index: Vec<i64> = (0..n as i64).collect();
    let row_index_d = stream.clone_htod(&row_index)?;
    let n_u64 = n as u64;
    let ncols_u32 = MEMW_REGISTER_NCOLS as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&keys_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&is_read_d)
            .arg(&row_index_d)
            .arg(&old_value_d)
            .arg(&old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok(buf)
}

// -----------------------------------------------------------------------------
// MEMW_R device walk + fill (no host walk, no old_* upload).
//
// The register memory-model walk runs on device (`trace_walk::walk_core`), and its
// resident `(old_value, old_ts)` feed the MEMW_R fill on the same stream — the
// recovered predecessors never touch the host. Input is the FULL register-access
// stream (every access advances the per-address timeline, so timestamp-only writes
// such as the implicit PC bump must be present); `row_index[i] < 0` marks an access
// that advances the timeline but emits no MEMW_R row (skipped by the fill kernel).
// `reg_addr` is both the walk bucket key and the fill's address column (register
// word address `< nbins`). See `reports/tracegen/GPU-WALK-PLAN.md`.
// -----------------------------------------------------------------------------

/// Device-resident register walk **and** MEMW_R fill. Recovers each access's
/// `(old_value, old_ts)` on device then fills the 10 MEMW_R columns row-major,
/// returning the residency-ready `[row*NCOLS+col]` buffer. `init_value[b]` seeds
/// bucket `b` at timestamp `init_ts` (the continuation register state).
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_and_fill_memw_register(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    row_index: &[i64],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let buf = walk_and_fill_memw_register_on(
        &stream, reg_addr, ts, value, is_read, row_index, init_value, init_ts, nbins, num_rows,
    )?;
    stream.synchronize()?;
    Ok(buf)
}

/// Host-returning wrapper over [`gpu_walk_and_fill_memw_register`] for byte-parity
/// tests against the CPU walk + fill.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_and_fill_memw_register_host(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    row_index: &[i64],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let buf = walk_and_fill_memw_register_on(
        &stream, reg_addr, ts, value, is_read, row_index, init_value, init_ts, nbins, num_rows,
    )?;
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

#[allow(clippy::too_many_arguments)]
fn walk_and_fill_memw_register_on(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    row_index: &[i64],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let n = reg_addr.len();
    debug_assert_eq!(ts.len(), n);
    debug_assert_eq!(value.len(), n);
    debug_assert_eq!(is_read.len(), n);
    debug_assert_eq!(row_index.len(), n);
    debug_assert_eq!(init_value.len(), nbins as usize);
    let be = backend()?;
    let mut buf = stream.alloc_zeros::<u64>(num_rows * MEMW_REGISTER_NCOLS)?;
    if n == 0 {
        return Ok(buf);
    }

    // Upload the access SoA once; `reg_addr` doubles as the walk bucket key and the
    // fill address column, and `ts`/`value` are shared by walk and fill.
    let keys_d = stream.clone_htod(reg_addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let init_value_d = stream.clone_htod(init_value)?;

    // Device walk → resident (old_value, old_ts), consumed in place by the fill.
    let (old_value_d, old_ts_d) = crate::trace_walk::walk_core(
        be,
        stream,
        &keys_d,
        &ts_d,
        &value_d,
        &init_value_d,
        init_ts,
        n,
        nbins,
    )?;

    let is_read_d = stream.clone_htod(is_read)?;
    let row_index_d = stream.clone_htod(row_index)?;
    let n_u64 = n as u64;
    let ncols_u32 = MEMW_REGISTER_NCOLS as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&keys_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&is_read_d)
            .arg(&row_index_d)
            .arg(&old_value_d)
            .arg(&old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok(buf)
}

/// P2: FULLY-RESIDENT register MEMW_R build — emit the register accesses on device from the resident
/// cpu_op fields (P1 `emit_register_accesses_dev`), walk them (`walk_core`), and fill the MEMW_R
/// table, all on one stream with NO host collection or per-access upload (only the cpu_op SoA + the
/// tiny `init_value[nbins]` are uploaded). Returns the residency-ready `[row*NCOLS+col]` buffer.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_fill_memw_register_resident(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;

    let (reg_addr_d, ts_d, value_d, is_read_d, row_index_d, total) =
        crate::trace_walk::emit_register_accesses_dev(
            be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n,
        )?;

    let mut buf = stream.alloc_zeros::<u64>(num_rows * MEMW_REGISTER_NCOLS)?;
    if total == 0 {
        stream.synchronize()?;
        return Ok(buf);
    }
    let init_value_d = stream.clone_htod(init_value)?;
    let (old_value_d, old_ts_d) = crate::trace_walk::walk_core(
        be, &stream, &reg_addr_d, &ts_d, &value_d, &init_value_d, init_ts, total, nbins,
    )?;
    let n_u64 = total as u64;
    let ncols_u32 = MEMW_REGISTER_NCOLS as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&reg_addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&is_read_d)
            .arg(&row_index_d)
            .arg(&old_value_d)
            .arg(&old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    stream.synchronize()?;
    Ok(buf)
}

/// P1-ecall: resident register walk that INJECTS extra (host-captured) register accesses as
/// NON-EMITTING timeline events. Emits the regular accesses on device from the cpu_op fields
/// (`emit_register_accesses_dev`), concatenates the injected `(reg_addr, ts, value, is_read)` with
/// `row_index = -1` (they advance the per-register timeline so REGULAR rows get correct old_ts across
/// the ecall interleaving, but emit no MEMW_R row — Option Z), walks the combined stream, and fills.
/// Byte-identical to `gpu_walk_and_fill_memw_register_host` fed the same combined stream. The injected
/// accesses' upload is tiny (ecall counts are small). `num_rows` is sized for the regular emitting rows.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_fill_memw_register_resident_injected(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    inj_reg_addr: &[u32],
    inj_ts: &[u64],
    inj_value: &[u64],
    inj_is_read: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let m = inj_reg_addr.len();
    debug_assert!(
        [inj_ts.len(), inj_value.len(), inj_is_read.len()].iter().all(|&l| l == m),
        "injected access arrays must be equal length"
    );
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;

    let (reg_addr_e, ts_e, value_e, is_read_e, row_index_e, total_e) =
        crate::trace_walk::emit_register_accesses_dev(
            be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n,
        )?;

    let total = total_e + m;
    let mut buf = stream.alloc_zeros::<u64>(num_rows * MEMW_REGISTER_NCOLS)?;
    if total == 0 {
        stream.synchronize()?;
        return Ok(buf);
    }

    // Combined access arrays: [emitted regular | injected ecall (row_index = -1)].
    let mut reg_addr = stream.alloc_zeros::<u32>(total)?;
    let mut ts = stream.alloc_zeros::<u64>(total)?;
    let mut value = stream.alloc_zeros::<u64>(total)?;
    let mut is_read = stream.alloc_zeros::<u8>(total)?;
    let mut row_index = stream.alloc_zeros::<i64>(total)?;
    if total_e > 0 {
        stream.memcpy_dtod(&reg_addr_e.slice(0..total_e), &mut reg_addr.slice_mut(0..total_e))?;
        stream.memcpy_dtod(&ts_e.slice(0..total_e), &mut ts.slice_mut(0..total_e))?;
        stream.memcpy_dtod(&value_e.slice(0..total_e), &mut value.slice_mut(0..total_e))?;
        stream.memcpy_dtod(&is_read_e.slice(0..total_e), &mut is_read.slice_mut(0..total_e))?;
        stream.memcpy_dtod(&row_index_e.slice(0..total_e), &mut row_index.slice_mut(0..total_e))?;
    }
    if m > 0 {
        stream.memcpy_htod(inj_reg_addr, &mut reg_addr.slice_mut(total_e..total))?;
        stream.memcpy_htod(inj_ts, &mut ts.slice_mut(total_e..total))?;
        stream.memcpy_htod(inj_value, &mut value.slice_mut(total_e..total))?;
        stream.memcpy_htod(inj_is_read, &mut is_read.slice_mut(total_e..total))?;
        stream.memcpy_htod(&vec![-1i64; m], &mut row_index.slice_mut(total_e..total))?;
    }

    let init_value_d = stream.clone_htod(init_value)?;
    let (old_value_d, old_ts_d) = crate::trace_walk::walk_core(
        be, &stream, &reg_addr, &ts, &value, &init_value_d, init_ts, total, nbins,
    )?;
    let n_u64 = total as u64;
    let ncols_u32 = MEMW_REGISTER_NCOLS as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&reg_addr)
            .arg(&ts)
            .arg(&value)
            .arg(&is_read)
            .arg(&row_index)
            .arg(&old_value_d)
            .arg(&old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    stream.synchronize()?;
    Ok(buf)
}

/// Host-returning wrapper over [`gpu_walk_fill_memw_register_resident_injected`] for parity tests.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_fill_memw_register_resident_injected_host(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    inj_reg_addr: &[u32],
    inj_ts: &[u64],
    inj_value: &[u64],
    inj_is_read: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let buf = gpu_walk_fill_memw_register_resident_injected(
        packed, rv1, rv2, rvd, next_pc, inj_reg_addr, inj_ts, inj_value, inj_is_read, init_value,
        init_ts, nbins, num_rows,
    )?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// P1-ecall: fully-resident register MEMW_R build that INTERLEAVES the ecall accesses at their op's
/// timeline position (via `emit_register_accesses_with_ecall_dev`), so the regular MEMW_R rows get
/// correct `old_ts`/`old_value` across the ecall interleaving. The ecall accesses are non-emitting
/// (their MEMW rows are produced on CPU per Option Z). Only the tiny ecall arrays + cpu_op SoA +
/// `init_value` are uploaded. `num_rows` is sized for the regular emitting rows.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_fill_memw_register_resident_ecall(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    ecall_op_index: &[u32],
    ecall_reg_addr: &[u32],
    ecall_ts: &[u64],
    ecall_value: &[u64],
    ecall_is_read: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;

    let (reg_addr_d, ts_d, value_d, is_read_d, row_index_d, total) =
        crate::trace_walk::emit_register_accesses_with_ecall_dev(
            be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n, ecall_op_index, ecall_reg_addr,
            ecall_ts, ecall_value, ecall_is_read,
        )?;

    let mut buf = stream.alloc_zeros::<u64>(num_rows * MEMW_REGISTER_NCOLS)?;
    if total == 0 {
        stream.synchronize()?;
        return Ok(buf);
    }
    let init_value_d = stream.clone_htod(init_value)?;
    let (old_value_d, old_ts_d) = crate::trace_walk::walk_core(
        be, &stream, &reg_addr_d, &ts_d, &value_d, &init_value_d, init_ts, total, nbins,
    )?;
    let n_u64 = total as u64;
    let ncols_u32 = MEMW_REGISTER_NCOLS as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&reg_addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&is_read_d)
            .arg(&row_index_d)
            .arg(&old_value_d)
            .arg(&old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    stream.synchronize()?;
    Ok(buf)
}

/// Host-returning wrapper over [`gpu_walk_fill_memw_register_resident_ecall`] for parity tests.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_fill_memw_register_resident_ecall_host(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    ecall_op_index: &[u32],
    ecall_reg_addr: &[u32],
    ecall_ts: &[u64],
    ecall_value: &[u64],
    ecall_is_read: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let buf = gpu_walk_fill_memw_register_resident_ecall(
        packed, rv1, rv2, rvd, next_pc, ecall_op_index, ecall_reg_addr, ecall_ts, ecall_value,
        ecall_is_read, init_value, init_ts, nbins, num_rows,
    )?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// Host-returning wrapper over [`gpu_walk_fill_memw_register_resident`] for parity tests.
#[allow(clippy::too_many_arguments)]
pub fn gpu_walk_fill_memw_register_resident_host(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let buf = gpu_walk_fill_memw_register_resident(
        packed, rv1, rv2, rvd, next_pc, init_value, init_ts, nbins, num_rows,
    )?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}
