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

/// DVRM table column count (`prover::tables::dvrm::cols::NUM_COLUMNS`).
pub const NUM_DVRM_COLS: usize = 34;

/// MEMW_R table column count (`prover::tables::memw_register::cols::NUM_COLUMNS`).
pub const NUM_MEMW_REGISTER_COLS: usize = 10;

/// MEMW_R per-op input stride (must match `trace_memw_register.cu` `MR_STRIDE`).
pub const MEMW_REGISTER_STRIDE: usize = 8;

/// LOAD table column count (`prover::tables::load::cols::NUM_COLUMNS`).
pub const NUM_LOAD_COLS: usize = 18;
/// LOAD per-op input stride (must match `trace_ldst.cu` `LOAD_STRIDE`).
pub const LOAD_STRIDE: usize = 12;

/// STORE table column count (`prover::tables::store::cols::NUM_COLUMNS`).
pub const NUM_STORE_COLS: usize = 16;
/// STORE per-op input stride (must match `trace_ldst.cu` `STORE_STRIDE`).
pub const STORE_STRIDE: usize = 4;

/// MEMW_A table column count (`prover::tables::memw_aligned::cols::NUM_COLUMNS`).
pub const NUM_MEMW_ALIGNED_COLS: usize = 29;
/// MEMW_A per-op input stride (must match `trace_memw.cu` `MA_STRIDE`).
pub const MEMW_ALIGNED_STRIDE: usize = 22;

/// MEMW table column count (`prover::tables::memw::cols::NUM_COLUMNS`).
pub const NUM_MEMW_COLS: usize = 49;
/// MEMW per-op input stride (must match `trace_memw.cu` `MW_STRIDE`).
pub const MEMW_STRIDE: usize = 29;

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

/// Build the MEMW_R (register memory) trace table on device. No dedup (one row
/// per op). `input` is interleaved with stride `MEMW_REGISTER_STRIDE`:
/// `[base_address, timestamp, value0, value1, old0, old1, old_timestamp0,
/// is_read]` per op. Padding rows (r >= n) stay zero. Mirrors
/// `generate_memw_register_trace`.
pub fn gpu_build_memw_register_trace(
    input: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(input.len(), n * MEMW_REGISTER_STRIDE);
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let in_d = stream.clone_htod(input)?;
    let mut cols = stream.alloc_zeros::<u64>(NUM_MEMW_REGISTER_COLS * nrows)?;

    let n_u64 = n as u64;
    let nrows_u64 = nrows as u64;
    let cfg = LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.trace_memw_register_kernel)
            .arg(&in_d)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols: NUM_MEMW_REGISTER_COLS,
        nrows,
    })
}

/// Shared launcher for the per-op memory fills (LOAD/STORE/MEMW_A/MEMW). Each
/// kernel reads an interleaved `input` (stride bytes per op) and writes a
/// column-major `ncols × nrows` buffer; padding rows (r >= n) stay zero.
fn launch_interleaved_fill(
    kernel: &cudarc::driver::CudaFunction,
    input: &[u64],
    stride: usize,
    n: usize,
    nrows: usize,
    ncols: usize,
) -> Result<DeviceMainCols> {
    assert_eq!(input.len(), n * stride);
    assert!(nrows >= n, "nrows must be >= n");

    let be = backend()?;
    let stream = be.next_stream();

    let in_d = stream.clone_htod(input)?;
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
            .arg(&in_d)
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

/// Build the LOAD trace on device. `input` interleaved, stride `LOAD_STRIDE`:
/// `[base_address, timestamp, width, signed, res0..res7]`. Mirrors
/// `generate_load_trace`.
pub fn gpu_build_load_trace(input: &[u64], n: usize, nrows: usize) -> Result<DeviceMainCols> {
    let be = backend()?;
    launch_interleaved_fill(
        &be.trace_load_kernel,
        input,
        LOAD_STRIDE,
        n,
        nrows,
        NUM_LOAD_COLS,
    )
}

/// Build the STORE trace on device. `input` interleaved, stride `STORE_STRIDE`:
/// `[base_address, timestamp, value, write_flags]`. Mirrors
/// `generate_store_trace`.
pub fn gpu_build_store_trace(input: &[u64], n: usize, nrows: usize) -> Result<DeviceMainCols> {
    let be = backend()?;
    launch_interleaved_fill(
        &be.trace_store_kernel,
        input,
        STORE_STRIDE,
        n,
        nrows,
        NUM_STORE_COLS,
    )
}

/// Build the MEMW_A (aligned) trace on device. `input` interleaved, stride
/// `MEMW_ALIGNED_STRIDE`: `[is_register, base_address, value0..value7,
/// timestamp, width, old0..old7, old_timestamp0, is_read]`. Mirrors
/// `generate_memw_aligned_trace`.
pub fn gpu_build_memw_aligned_trace(
    input: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    let be = backend()?;
    launch_interleaved_fill(
        &be.trace_memw_aligned_kernel,
        input,
        MEMW_ALIGNED_STRIDE,
        n,
        nrows,
        NUM_MEMW_ALIGNED_COLS,
    )
}

/// Build the MEMW (general) trace on device. `input` interleaved, stride
/// `MEMW_STRIDE`: `[is_register, base_address, value0..value7, timestamp,
/// width, old0..old7, old_timestamp0..old_timestamp7, is_read]`. Mirrors
/// `generate_memw_trace`.
pub fn gpu_build_memw_trace(input: &[u64], n: usize, nrows: usize) -> Result<DeviceMainCols> {
    let be = backend()?;
    launch_interleaved_fill(
        &be.trace_memw_kernel,
        input,
        MEMW_STRIDE,
        n,
        nrows,
        NUM_MEMW_COLS,
    )
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

/// Build the DVRM trace table on device from deduped ops. SoA over `n` unique
/// ops: `n_num`, `d_den`, `flags` (bit0=signed), `mu_q`, `mu_r`. Padding rows
/// stay zero.
pub fn gpu_build_dvrm_trace(
    n_num: &[u64],
    d_den: &[u64],
    flags: &[u64],
    mu_q: &[u64],
    mu_r: &[u64],
    n: usize,
    nrows: usize,
) -> Result<DeviceMainCols> {
    build_alu5(
        &backend()?.trace_dvrm_kernel,
        n_num,
        d_den,
        flags,
        mu_q,
        mu_r,
        n,
        nrows,
        NUM_DVRM_COLS,
    )
}

/// Unique keys + accumulated multiplicity counters from a GPU hash group-by.
pub struct DedupResult {
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub c: Vec<u64>,
    pub mu0: Vec<u64>,
    pub mu1: Vec<u64>,
}

/// Device-resident dedup output: unique keys + counters stay in VRAM. Only the
/// first `n_unique` entries of each (capacity-`M`) buffer are valid.
pub struct DeviceDedup {
    pub a: CudaSlice<u64>,
    pub b: CudaSlice<u64>,
    pub c: CudaSlice<u64>,
    pub mu0: CudaSlice<u64>,
    pub mu1: CudaSlice<u64>,
    pub n_unique: usize,
}

/// GPU dedup (hash group-by) of key triples `(a, b, c)`, keeping the unique
/// keys + summed counters resident in VRAM. `sel[i]` selects the counter
/// (0 -> mu0, 1 -> mu1); pass an all-zero `sel` for single-counter tables.
pub fn gpu_dedup_device(a: &[u64], b: &[u64], c: &[u64], sel: &[u64]) -> Result<DeviceDedup> {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(sel.len(), n);
    // Load factor <= 0.5 guarantees a free slot (probe loop terminates).
    let m = (2 * n.max(1)).next_power_of_two().max(4);
    let be = backend()?;
    let stream = be.next_stream();

    let a_d = stream.clone_htod(a)?;
    let b_d = stream.clone_htod(b)?;
    let c_d = stream.clone_htod(c)?;
    let sel_d = stream.clone_htod(sel)?;

    let mut slot = stream.alloc_zeros::<u64>(m)?;
    let mut mu0 = stream.alloc_zeros::<u64>(m)?;
    let mut mu1 = stream.alloc_zeros::<u64>(m)?;

    let m_u64 = m as u64;
    let n_u64 = n as u64;
    let blk = |x: usize| LaunchConfig {
        grid_dim: ((x.max(1) as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&be.dedup_init)
            .arg(&mut slot)
            .arg(&mut mu0)
            .arg(&mut mu1)
            .arg(&m_u64)
            .launch(blk(m))?;
    }
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.dedup_insert)
                .arg(&a_d)
                .arg(&b_d)
                .arg(&c_d)
                .arg(&sel_d)
                .arg(&n_u64)
                .arg(&m_u64)
                .arg(&mut slot)
                .arg(&mut mu0)
                .arg(&mut mu1)
                .launch(blk(n))?;
        }
    }

    let mut out_a = stream.alloc_zeros::<u64>(m)?;
    let mut out_b = stream.alloc_zeros::<u64>(m)?;
    let mut out_c = stream.alloc_zeros::<u64>(m)?;
    let mut out_mu0 = stream.alloc_zeros::<u64>(m)?;
    let mut out_mu1 = stream.alloc_zeros::<u64>(m)?;
    let mut out_count = stream.alloc_zeros::<u64>(1)?;
    unsafe {
        stream
            .launch_builder(&be.dedup_compact)
            .arg(&slot)
            .arg(&mu0)
            .arg(&mu1)
            .arg(&m_u64)
            .arg(&a_d)
            .arg(&b_d)
            .arg(&c_d)
            .arg(&mut out_a)
            .arg(&mut out_b)
            .arg(&mut out_c)
            .arg(&mut out_mu0)
            .arg(&mut out_mu1)
            .arg(&mut out_count)
            .launch(blk(m))?;
    }
    stream.synchronize()?;
    let n_unique = stream.clone_dtoh(&out_count)?[0] as usize;

    Ok(DeviceDedup {
        a: out_a,
        b: out_b,
        c: out_c,
        mu0: out_mu0,
        mu1: out_mu1,
        n_unique,
    })
}

/// Host-returning dedup (for tests / debugging). See [`gpu_dedup_device`].
pub fn gpu_dedup(a: &[u64], b: &[u64], c: &[u64], sel: &[u64]) -> Result<DedupResult> {
    let dd = gpu_dedup_device(a, b, c, sel)?;
    let nu = dd.n_unique;
    let be = backend()?;
    let stream = be.next_stream();
    let pull = |buf: &CudaSlice<u64>| -> Result<Vec<u64>> {
        Ok(stream.clone_dtoh(buf)?[..nu].to_vec())
    };
    Ok(DedupResult {
        a: pull(&dd.a)?,
        b: pull(&dd.b)?,
        c: pull(&dd.c)?,
        mu0: pull(&dd.mu0)?,
        mu1: pull(&dd.mu1)?,
    })
}

// Launch config for a per-row kernel covering `nrows` rows.
fn rows_cfg(nrows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((nrows.max(1) as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Device-input fill for 4-input ALU kernels `(a, b, c, mu, n, nrows, cols)`
/// (lt/eq/bytewise). Reads already-resident device buffers — no H2D.
fn fill_dev4(
    kernel: &cudarc::driver::CudaFunction,
    a: &CudaSlice<u64>,
    b: &CudaSlice<u64>,
    c: &CudaSlice<u64>,
    mu: &CudaSlice<u64>,
    n: usize,
    nrows: usize,
    ncols: usize,
) -> Result<DeviceMainCols> {
    let be = backend()?;
    let stream = be.next_stream();
    let mut cols = stream.alloc_zeros::<u64>(ncols * nrows)?;
    let (n_u64, nrows_u64) = (n as u64, nrows as u64);
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(mu)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(rows_cfg(nrows))?;
    }
    stream.synchronize()?;
    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols,
        nrows,
    })
}

/// Device-input fill for 5-input ALU kernels `(a, b, c, mu0, mu1, n, nrows,
/// cols)` (mul/dvrm). Reads already-resident device buffers — no H2D.
#[allow(clippy::too_many_arguments)]
fn fill_dev5(
    kernel: &cudarc::driver::CudaFunction,
    a: &CudaSlice<u64>,
    b: &CudaSlice<u64>,
    c: &CudaSlice<u64>,
    mu0: &CudaSlice<u64>,
    mu1: &CudaSlice<u64>,
    n: usize,
    nrows: usize,
    ncols: usize,
) -> Result<DeviceMainCols> {
    let be = backend()?;
    let stream = be.next_stream();
    let mut cols = stream.alloc_zeros::<u64>(ncols * nrows)?;
    let (n_u64, nrows_u64) = (n as u64, nrows as u64);
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(mu0)
            .arg(mu1)
            .arg(&n_u64)
            .arg(&nrows_u64)
            .arg(&mut cols)
            .launch(rows_cfg(nrows))?;
    }
    stream.synchronize()?;
    Ok(DeviceMainCols {
        buf: Arc::new(cols),
        ncols,
        nrows,
    })
}

// --- Fused dedup + fill: GPU group-by then device-resident column fill, no
// host round-trip. Inputs are the raw (un-deduped) op SoA; `sel` is all-zero
// for single-counter tables, or the counter selector for dual ones. ---

/// LT (single counter). Key = (lhs, rhs, flags); `sel` all-zero.
pub fn gpu_build_lt_trace_deduped(
    a: &[u64],
    b: &[u64],
    c: &[u64],
    sel: &[u64],
) -> Result<DeviceMainCols> {
    let dd = gpu_dedup_device(a, b, c, sel)?;
    let nrows = dd.n_unique.next_power_of_two().max(4);
    fill_dev4(
        &backend()?.trace_lt_kernel,
        &dd.a,
        &dd.b,
        &dd.c,
        &dd.mu0,
        dd.n_unique,
        nrows,
        NUM_LT_COLS,
    )
}

/// EQ (single counter). Key = (a, b, flags=invert); `sel` all-zero.
pub fn gpu_build_eq_trace_deduped(
    a: &[u64],
    b: &[u64],
    c: &[u64],
    sel: &[u64],
) -> Result<DeviceMainCols> {
    let dd = gpu_dedup_device(a, b, c, sel)?;
    let nrows = dd.n_unique.next_power_of_two().max(4);
    fill_dev4(
        &backend()?.trace_eq_kernel,
        &dd.a,
        &dd.b,
        &dd.c,
        &dd.mu0,
        dd.n_unique,
        nrows,
        NUM_EQ_COLS,
    )
}

/// BYTEWISE (single counter). Key = (a, b, op); `sel` all-zero.
pub fn gpu_build_bytewise_trace_deduped(
    a: &[u64],
    b: &[u64],
    c: &[u64],
    sel: &[u64],
) -> Result<DeviceMainCols> {
    let dd = gpu_dedup_device(a, b, c, sel)?;
    let nrows = dd.n_unique.next_power_of_two().max(4);
    fill_dev4(
        &backend()?.trace_bytewise_kernel,
        &dd.a,
        &dd.b,
        &dd.c,
        &dd.mu0,
        dd.n_unique,
        nrows,
        NUM_BYTEWISE_COLS,
    )
}

/// MUL (dual counters). Key = (lhs, rhs, flags); `sel` = wants_hi.
pub fn gpu_build_mul_trace_deduped(
    a: &[u64],
    b: &[u64],
    c: &[u64],
    sel: &[u64],
) -> Result<DeviceMainCols> {
    let dd = gpu_dedup_device(a, b, c, sel)?;
    let nrows = dd.n_unique.next_power_of_two().max(4);
    fill_dev5(
        &backend()?.trace_mul_kernel,
        &dd.a,
        &dd.b,
        &dd.c,
        &dd.mu0,
        &dd.mu1,
        dd.n_unique,
        nrows,
        NUM_MUL_COLS,
    )
}

/// DVRM (dual counters). Key = (n, d, flags=signed); `sel` = wants_remainder.
pub fn gpu_build_dvrm_trace_deduped(
    a: &[u64],
    b: &[u64],
    c: &[u64],
    sel: &[u64],
) -> Result<DeviceMainCols> {
    let dd = gpu_dedup_device(a, b, c, sel)?;
    let nrows = dd.n_unique.next_power_of_two().max(4);
    fill_dev5(
        &backend()?.trace_dvrm_kernel,
        &dd.a,
        &dd.b,
        &dd.c,
        &dd.mu0,
        &dd.mu1,
        dd.n_unique,
        nrows,
        NUM_DVRM_COLS,
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
