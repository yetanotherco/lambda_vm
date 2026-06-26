//! GPU VM trace generation.
//!
//! Builds main-trace tables directly on device, column-major, returning a
//! resident [`DeviceMainCols`] handle that feeds the LDE without a host
//! round-trip. First table: CPU (see `kernels/trace_cpu.cu`).
//!
//! The instruction *decoder* stays on the host (run once per program); this
//! layer only does the per-cycle `from_log` + column fill on device. Callers
//! pass flat `u64` arrays so this crate stays VM-agnostic.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// CPU table column count (must match `prover::tables::cpu::cols::NUM_COLUMNS`).
pub const NUM_CPU_COLS: usize = 38;

/// LT table column count (must match `prover::tables::lt::cols::NUM_COLUMNS`).
pub const NUM_LT_COLS: usize = 17;

/// EQ table column count (`prover::tables::eq::cols::NUM_COLUMNS`).
pub const NUM_EQ_COLS: usize = 12;

/// BYTEWISE table column count (`prover::tables::bytewise::cols::NUM_COLUMNS`).
pub const NUM_BYTEWISE_COLS: usize = 26;

/// SHIFT table column count (`prover::tables::shift::cols::NUM_COLUMNS`).
pub const NUM_SHIFT_COLS: usize = 29;

/// MUL table column count (`prover::tables::mul::cols::NUM_COLUMNS`).
pub const NUM_MUL_COLS: usize = 26;

/// `PackedDecode` stride: u64s per program PC (must match `trace_cpu.cu`).
pub const DEC_STRIDE: usize = 8;

// PackedDecode field offsets within each DEC_STRIDE-u64 record.
pub const DEC_FLAGS: usize = 0;
pub const DEC_RS1: usize = 1;
pub const DEC_RS2: usize = 2;
pub const DEC_RD: usize = 3;
pub const DEC_HIL: usize = 4;
pub const DEC_ALU_FLAGS: usize = 5;
pub const DEC_MEM_FLAGS: usize = 6;
pub const DEC_IMM: usize = 7;

// Flag bit positions inside `DEC_FLAGS`.
pub const F_READ_REGISTER1: u32 = 0;
pub const F_READ_REGISTER2: u32 = 1;
pub const F_WRITE_REGISTER: u32 = 2;
pub const F_WORD_INSTR: u32 = 3;
pub const F_ALU: u32 = 4;
pub const F_ADD: u32 = 5;
pub const F_SUB: u32 = 6;
pub const F_MEMORY: u32 = 7;
pub const F_BRANCH: u32 = 8;
pub const F_ECALL: u32 = 9;

/// A pre-LDE, column-major trace block resident in VRAM.
///
/// Column `c` occupies `buf[c*nrows .. c*nrows + nrows]`; values are canonical
/// Goldilocks u64s. This is the pre-LDE sibling of [`crate::lde::GpuLdeBase`]:
/// `nrows` is the (padded, power-of-two) trace length the LDE takes as input.
#[derive(Clone)]
pub struct DeviceMainCols {
    pub buf: Arc<CudaSlice<u64>>,
    pub ncols: usize,
    pub nrows: usize,
}

// `CudaSlice` implements none of these; provide structural impls so a handle can
// live on `TraceTable` (which derives Debug/PartialEq/Eq). Equality is by shape
// + buffer identity (Arc pointer), which is all callers need.
impl std::fmt::Debug for DeviceMainCols {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMainCols")
            .field("ncols", &self.ncols)
            .field("nrows", &self.nrows)
            .finish_non_exhaustive()
    }
}
impl PartialEq for DeviceMainCols {
    fn eq(&self, other: &Self) -> bool {
        self.ncols == other.ncols
            && self.nrows == other.nrows
            && Arc::ptr_eq(&self.buf, &other.buf)
    }
}
impl Eq for DeviceMainCols {}

impl DeviceMainCols {
    /// Upload a host column-major buffer (`ncols * nrows`) to device. For tests
    /// / bring-up that build columns on host and want a resident handle.
    pub fn upload(flat_colmajor: &[u64], ncols: usize, nrows: usize) -> Result<Self> {
        assert_eq!(flat_colmajor.len(), ncols * nrows, "buffer must be ncols*nrows");
        let be = backend()?;
        let stream = be.next_stream();
        let buf = stream.clone_htod(flat_colmajor)?;
        Ok(Self {
            buf: Arc::new(buf),
            ncols,
            nrows,
        })
    }

    /// Copy the full column-major buffer back to host. For parity tests /
    /// debugging — the resident path never calls this.
    pub fn to_host(&self) -> Result<Vec<u64>> {
        let be = backend()?;
        let stream = be.next_stream();
        let host = stream.clone_dtoh(self.buf.as_ref())?;
        Ok(host)
    }
}

/// Build the CPU trace table on device, column-major.
///
/// - `logs`: `5 * n` u64s, `[current_pc, next_pc, src1_val, src2_val, dst_val]`
///   per executed instruction.
/// - `decode`: `DEC_STRIDE * n_pc` u64s — the `PackedDecode` array indexed by
///   `(pc - text_base) >> 1` (RISC-V is 2-byte aligned). Built once on host.
/// - `row_offset`: global index of this chunk's first row (0 for a single
///   un-chunked table; `chunk_index * max_rows` otherwise). Timestamps are
///   global: `ts = 4*(row_offset + r) + 4`.
/// - `nrows`: padded power-of-two length of THIS chunk
///   (`n.next_power_of_two().max(4)`).
pub fn gpu_build_cpu_trace(
    logs: &[u64],
    decode: &[u64],
    text_base: u64,
    row_offset: u64,
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(logs.len(), 5 * n, "logs must be 5*n u64s");
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let logs_dev = stream.clone_htod(logs)?;
    let decode_dev = stream.clone_htod(decode)?;
    let mut cols = stream.alloc_zeros::<u64>(NUM_CPU_COLS * nrows)?;

    let n_u64 = n as u64;
    let nrows_u64 = nrows as u64;
    let cfg = LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.trace_cpu_kernel)
            .arg(&logs_dev)
            .arg(&decode_dev)
            .arg(&text_base)
            .arg(&row_offset)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols: NUM_CPU_COLS,
        nrows,
    })
}

/// Build the LT trace table on device from already-deduped operations,
/// column-major. Inputs are SoA over `n` unique ops: `lhs`, `rhs`, `flags`
/// (bit0=signed, bit1=invert), `mult` (summed multiplicity). `nrows` is the
/// padded power-of-two row count (`n.next_power_of_two().max(4)`); padding rows
/// stay zero.
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_lt_trace(
    lhs: &[u64],
    rhs: &[u64],
    flags: &[u64],
    mult: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(lhs.len(), n);
    assert_eq!(rhs.len(), n);
    assert_eq!(flags.len(), n);
    assert_eq!(mult.len(), n);
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let lhs_d = stream.clone_htod(lhs)?;
    let rhs_d = stream.clone_htod(rhs)?;
    let flags_d = stream.clone_htod(flags)?;
    let mult_d = stream.clone_htod(mult)?;
    let mut cols = stream.alloc_zeros::<u64>(NUM_LT_COLS * nrows)?;

    let n_u64 = n as u64;
    let nrows_u64 = nrows as u64;
    let cfg = LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.trace_lt_kernel)
            .arg(&lhs_d)
            .arg(&rhs_d)
            .arg(&flags_d)
            .arg(&mult_d)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols: NUM_LT_COLS,
        nrows,
    })
}

/// Build the EQ trace table on device from deduped ops. SoA over `n` unique
/// ops: `a`, `b`, `flags` (bit0=invert), `mult`. Padding rows stay zero.
pub fn gpu_build_eq_trace(
    a: &[u64],
    b: &[u64],
    flags: &[u64],
    mult: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    build_alu4(&backend()?.trace_eq_kernel, a, b, flags, mult, n, nrows, NUM_EQ_COLS)
}

/// Build the BYTEWISE trace table on device from deduped ops. SoA over `n`
/// unique ops: `a`, `b`, `op` (AND=0/OR=1/XOR=2), `mult`. Padding rows stay zero.
pub fn gpu_build_bytewise_trace(
    a: &[u64],
    b: &[u64],
    op: &[u64],
    mult: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    build_alu4(
        &backend()?.trace_bytewise_kernel,
        a,
        b,
        op,
        mult,
        n,
        nrows,
        NUM_BYTEWISE_COLS,
    )
}

/// Build the SHIFT trace table on device. SHIFT does not dedup (one row per op,
/// μ=1). SoA over `n` ops: `value` (4 input halves packed into a u64),
/// `shift_amount` (full arg2), `flags` (bit0=direction, bit1=signed,
/// bit2=word_instr). Padding rows set ZBS=1 (kernel-side).
pub fn gpu_build_shift_trace(
    value: &[u64],
    shift_amount: &[u64],
    flags: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(value.len(), n);
    assert_eq!(shift_amount.len(), n);
    assert_eq!(flags.len(), n);
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let v_d = stream.clone_htod(value)?;
    let sa_d = stream.clone_htod(shift_amount)?;
    let f_d = stream.clone_htod(flags)?;
    let mut cols = stream.alloc_zeros::<u64>(NUM_SHIFT_COLS * nrows)?;

    let n_u64 = n as u64;
    let nrows_u64 = nrows as u64;
    let cfg = LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.trace_shift_kernel)
            .arg(&v_d)
            .arg(&sa_d)
            .arg(&f_d)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols: NUM_SHIFT_COLS,
        nrows,
    })
}

/// Build the MUL trace table on device from deduped ops. SoA over `n` unique
/// ops: `lhs`, `rhs`, `flags` (bit0=lhs_signed, bit1=rhs_signed), `mu_lo`,
/// `mu_hi`. Padding rows stay zero.
pub fn gpu_build_mul_trace(
    lhs: &[u64],
    rhs: &[u64],
    flags: &[u64],
    mu_lo: &[u64],
    mu_hi: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    build_alu5(
        &backend()?.trace_mul_kernel,
        lhs,
        rhs,
        flags,
        mu_lo,
        mu_hi,
        n,
        nrows,
        NUM_MUL_COLS,
    )
}

/// Shared launcher for kernels taking 5 SoA u64 inputs (a, b, c, d, e) — e.g.
/// MUL/DVRM with dual multiplicity counters. One thread per row.
#[allow(clippy::too_many_arguments)]
fn build_alu5(
    kernel: &cudarc::driver::CudaFunction,
    a: &[u64],
    b: &[u64],
    c: &[u64],
    d: &[u64],
    e: &[u64],
    n: usize,
    nrows: usize,
    ncols: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(d.len(), n);
    assert_eq!(e.len(), n);
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let a_d = stream.clone_htod(a)?;
    let b_d = stream.clone_htod(b)?;
    let c_d = stream.clone_htod(c)?;
    let d_d = stream.clone_htod(d)?;
    let e_d = stream.clone_htod(e)?;
    let mut cols = stream.alloc_zeros::<u64>(ncols * nrows)?;

    let n_u64 = n as u64;
    let nrows_u64 = nrows as u64;
    let cfg = LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&a_d)
            .arg(&b_d)
            .arg(&c_d)
            .arg(&d_d)
            .arg(&e_d)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols,
        nrows,
    })
}

/// Shared launcher for ALU kernels taking 4 SoA u64 inputs (a, b, c, mult) and
/// producing an `ncols * nrows` column-major buffer. One thread per row.
#[allow(clippy::too_many_arguments)]
fn build_alu4(
    kernel: &cudarc::driver::CudaFunction,
    a: &[u64],
    b: &[u64],
    c: &[u64],
    mult: &[u64],
    n: usize,
    nrows: usize,
    ncols: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(mult.len(), n);
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let a_d = stream.clone_htod(a)?;
    let b_d = stream.clone_htod(b)?;
    let c_d = stream.clone_htod(c)?;
    let mult_d = stream.clone_htod(mult)?;
    let mut cols = stream.alloc_zeros::<u64>(ncols * nrows)?;

    let n_u64 = n as u64;
    let nrows_u64 = nrows as u64;
    let cfg = LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&a_d)
            .arg(&b_d)
            .arg(&c_d)
            .arg(&mult_d)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols,
        nrows,
    })
}
