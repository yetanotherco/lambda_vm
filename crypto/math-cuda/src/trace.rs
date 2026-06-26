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
