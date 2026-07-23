//! On-GPU BITWISE multiplicity histogram (see `kernels/bitwise_hist.cu`).
//!
//! The prover's `BitwiseHistogram` is a dense `[num_rows * num_types]` u64 counter
//! array bumped by ~55M range-check lookups per ethrex proof — a cache-missing scatter
//! that dominates trace-build. This scatters the big sources on device (atomics into
//! `HIST_COPIES` replicated histograms to defuse contention, then reduced) and returns
//! the counter array for the host to merge into its histogram.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{Backend, backend};

/// Replicated histogram copies to defuse atomic contention on the hot ARE_BYTES bins.
/// Each uses `num_rows * num_types * 8` bytes (~80 MiB), so 32 ≈ 2.5 GiB of VRAM.
const HIST_COPIES: u32 = 32;

/// Per-op SoA fields for the in-walk source (`CpuOperation::collect_bitwise_ops`).
pub struct CpuOpFields<'a> {
    pub rs1: &'a [u8],
    pub rs2: &'a [u8],
    pub rd: &'a [u8],
    pub hil: &'a [u8],
    pub alu_flags: &'a [u8],
    pub mem_flags: &'a [u8],
    pub res: &'a [u64],
    pub word: &'a [u8],
}

/// Launch the in-walk (per-CPU-op) source into the replicated histogram `hist`.
fn scatter_cpu_ops(
    be: &Backend,
    stream: &Arc<CudaStream>,
    f: &CpuOpFields,
    num_rows: u64,
    stride: u64,
    hist: &mut CudaSlice<u64>,
) -> Result<()> {
    let n = f.rs1.len();
    if n == 0 {
        return Ok(());
    }
    let rs1_d = stream.clone_htod(f.rs1)?;
    let rs2_d = stream.clone_htod(f.rs2)?;
    let rd_d = stream.clone_htod(f.rd)?;
    let hil_d = stream.clone_htod(f.hil)?;
    let alu_d = stream.clone_htod(f.alu_flags)?;
    let mem_d = stream.clone_htod(f.mem_flags)?;
    let res_d = stream.clone_htod(f.res)?;
    let word_d = stream.clone_htod(f.word)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_cpu_ops)
            .arg(&n_u64)
            .arg(&rs1_d)
            .arg(&rs2_d)
            .arg(&rd_d)
            .arg(&hil_d)
            .arg(&alu_d)
            .arg(&mem_d)
            .arg(&res_d)
            .arg(&word_d)
            .arg(&num_rows)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok(())
}

/// Launch the MEMW_R source (one IS_HALF per row, keyed by the ts delta) into `hist`.
fn scatter_memw_reg(
    be: &Backend,
    stream: &Arc<CudaStream>,
    ts: &[u64],
    old_ts: &[u64],
    num_rows: u64,
    stride: u64,
    hist: &mut CudaSlice<u64>,
) -> Result<()> {
    let n = ts.len();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(old_ts.len(), n);
    let ts_d = stream.clone_htod(ts)?;
    let old_ts_d = stream.clone_htod(old_ts)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_memw_reg)
            .arg(&n_u64)
            .arg(&ts_d)
            .arg(&old_ts_d)
            .arg(&num_rows)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok(())
}

/// Device BITWISE histogram over the big sources: the in-walk per-op checks and the
/// MEMW_R ts-delta IS_HALF lookups (`memw_ts`/`memw_old_ts`, empty to skip). Scatters
/// both into `HIST_COPIES` replicated histograms (atomics), reduces, and returns the
/// `[num_rows * num_types]` counter array for the host to merge. Empty inputs → zeros.
pub fn gpu_bitwise_hist(
    cpu_ops: &CpuOpFields,
    memw_ts: &[u64],
    memw_old_ts: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    if cpu_ops.rs1.is_empty() && memw_ts.is_empty() {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }

    let nr = num_rows as u64;
    let stride = total as u64;
    // R replicated histograms (contention fix); each source scatters into copy `blk % R`.
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;

    scatter_cpu_ops(be, &stream, cpu_ops, nr, stride, &mut hist)?;
    scatter_memw_reg(be, &stream, memw_ts, memw_old_ts, nr, stride, &mut hist)?;

    // Reduce the R copies into `out`.
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_reduce)
            .arg(&stride)
            .arg(&HIST_COPIES)
            .arg(&hist)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}
