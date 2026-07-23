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

/// RESIDENT in_walk (from the device cpu_ops seam `packed`/`res`, NO upload) + memw_reg + page, into
/// one replicated histogram. Same result as [`gpu_bitwise_hist_resident_upload`], but the in-walk
/// source reads the resident `DeviceCpuOpsResident` buffers directly instead of re-uploading
/// `packed`+`res` (~64 MiB saved — the biggest source, 62% of bumps, becomes upload-free). memw_reg /
/// page are still host-fed here (their residency is the next step). `n` = cpu_op count.
#[allow(clippy::too_many_arguments)]
pub fn gpu_bitwise_hist_resident_devops(
    packed: &CudaSlice<u64>,
    res: &CudaSlice<u64>,
    n: usize,
    memw_ts: &[u64],
    memw_old_ts: &[u64],
    page_init: &[u8],
    page_fini: &[u8],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    if n == 0 && memw_ts.is_empty() && page_init.is_empty() {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    if n > 0 {
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_cpu_ops_packed)
                .arg(&n_u64)
                .arg(packed)
                .arg(res)
                .arg(&nr)
                .arg(&HIST_COPIES)
                .arg(&stride)
                .arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    scatter_memw_reg(be, &stream, memw_ts, memw_old_ts, nr, stride, &mut hist)?;
    if !page_init.is_empty() {
        let init_d = stream.clone_htod(page_init)?;
        let fini_d = stream.clone_htod(page_fini)?;
        let pn = page_init.len() as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_page)
                .arg(&pn)
                .arg(&init_d)
                .arg(&fini_d)
                .arg(&nr)
                .arg(&HIST_COPIES)
                .arg(&stride)
                .arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(page_init.len() as u32))?;
        }
    }
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

/// RESIDENT-KERNEL BITWISE histogram for the two big sources, driven from the packed decode:
/// in-walk (unpacked on device from `packed`+`res`, no 8-array host SoA rebuild) + MEMW_R ts-delta.
/// Uploads only `packed`+`res` (2 arrays) instead of the 8 `CpuOpFields` arrays, and unpacks the
/// decode fields on device via `bitwise_hist_cpu_ops_packed`. Returns the `[num_rows*num_types]`
/// counter array (same layout as [`gpu_bitwise_hist`]). This is the in-build wiring of the resident
/// in-walk source. Empty inputs → zeros.
#[allow(clippy::too_many_arguments)]
pub fn gpu_bitwise_hist_resident_upload(
    packed: &[u64],
    res: &[u64],
    memw_ts: &[u64],
    memw_old_ts: &[u64],
    page_init: &[u8],
    page_fini: &[u8],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    if packed.is_empty() && memw_ts.is_empty() && page_init.is_empty() {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    if !packed.is_empty() {
        let pk_d = stream.clone_htod(packed)?;
        let res_d = stream.clone_htod(res)?;
        let n_u64 = packed.len() as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_cpu_ops_packed)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&res_d)
                .arg(&nr)
                .arg(&HIST_COPIES)
                .arg(&stride)
                .arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(packed.len() as u32))?;
        }
    }
    scatter_memw_reg(be, &stream, memw_ts, memw_old_ts, nr, stride, &mut hist)?;
    if !page_init.is_empty() {
        let init_d = stream.clone_htod(page_init)?;
        let fini_d = stream.clone_htod(page_fini)?;
        let pn = page_init.len() as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_page)
                .arg(&pn)
                .arg(&init_d)
                .arg(&fini_d)
                .arg(&nr)
                .arg(&HIST_COPIES)
                .arg(&stride)
                .arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(page_init.len() as u32))?;
        }
    }
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

/// MEMW_R-source BITWISE histogram (P3): scatters one IS_HALF per EMITTING register row (ts delta)
/// via `bitwise_hist_memw_reg_masked`, reducing to the `[num_rows*num_types]` counter array. Reads
/// the full access stream + `row_index` (skips non-emitting). Bit-identical to the CPU
/// `collect_bitwise_from_memw_register` over the same rows. Empty input → zeros.
pub fn gpu_bitwise_hist_memw_reg_masked(
    ts: &[u64],
    old_ts: &[u64],
    row_index: &[i64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(ts.len(), old_ts.len());
    assert_eq!(ts.len(), row_index.len());
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    let n = ts.len();
    if n == 0 {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let n_u64 = n as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    let ts_d = stream.clone_htod(ts)?;
    let ot_d = stream.clone_htod(old_ts)?;
    let ri_d = stream.clone_htod(row_index)?;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_memw_reg_masked)
            .arg(&n_u64)
            .arg(&ts_d)
            .arg(&ot_d)
            .arg(&ri_d)
            .arg(&nr)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(&mut hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
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

/// MEMW_ALIGNED op-vec source (P4): one IS_HALF[base_low + mask] per aligned memw op, reduced to
/// the counter array. `aligned[i]` = the walk's aligned classify flag. Bit-identical to the CPU
/// `collect_bitwise_from_memw_aligned`.
pub fn gpu_bitwise_hist_memw_aligned(
    base: &[u64],
    width: &[u32],
    aligned: &[u32],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(base.len(), width.len());
    assert_eq!(base.len(), aligned.len());
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    let n = base.len();
    if n == 0 {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let n_u64 = n as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    let base_d = stream.clone_htod(base)?;
    let width_d = stream.clone_htod(width)?;
    let aligned_d = stream.clone_htod(aligned)?;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_memw_aligned)
            .arg(&n_u64)
            .arg(&base_d)
            .arg(&width_d)
            .arg(&aligned_d)
            .arg(&nr)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(&mut hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
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

/// PAGE-source BITWISE histogram: scatters one ARE_BYTES[init, fini] per touched byte on device
/// (`bitwise_hist_page`), reduces, and returns the `[num_rows*num_types]` counter array. `init`/
/// `fini` are the per-byte init/final values built on host (dense page read). Bit-identical to the
/// PAGE contribution of the CPU `collect_bitwise_from_page`. Empty input → zeros.
pub fn gpu_bitwise_hist_page_only(
    init: &[u8],
    fini: &[u8],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(init.len(), fini.len(), "page init/fini length");
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    let n = init.len();
    if n == 0 {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let n_u64 = n as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    let init_d = stream.clone_htod(init)?;
    let fini_d = stream.clone_htod(fini)?;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_page)
            .arg(&n_u64)
            .arg(&init_d)
            .arg(&fini_d)
            .arg(&nr)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(&mut hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
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

/// Single-source scaffold shared by the op-vec histogram wrappers: allocate the
/// `[num_rows*num_types]` output + the `HIST_COPIES` replicated scratch, run the
/// caller's `scatter` (upload inputs + launch its source kernel into the scratch),
/// reduce the copies, and download. `n == 0` short-circuits to a zero array.
/// The `scatter` closure receives `(be, stream, num_rows, num_copies, copy_stride, hist)`.
fn run_source<F>(num_rows: usize, num_types: usize, n: usize, scatter: F) -> Result<Vec<u64>>
where
    F: FnOnce(&Backend, &Arc<CudaStream>, u64, u32, u64, &mut CudaSlice<u64>) -> Result<()>,
{
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    if n == 0 {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    scatter(be, &stream, nr, HIST_COPIES, stride, &mut hist)?;
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

/// LT op-vec source: per `(lhs,rhs)` op, 2 Msb16 + 6 IS_HALF. Bit-identical to
/// `collect_bitwise_from_lt`. Returns the `[num_rows*num_types]` counter array.
pub fn gpu_bitwise_hist_lt(
    lhs: &[u64],
    rhs: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(lhs.len(), rhs.len());
    let n = lhs.len();
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let lhs_d = stream.clone_htod(lhs)?;
        let rhs_d = stream.clone_htod(rhs)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_lt)
                .arg(&n_u64)
                .arg(&lhs_d)
                .arg(&rhs_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// STORE op-vec source: 8 ARE_BYTES per op (bytes of `value`). Bit-identical to
/// `StoreOperation::collect_bitwise_ops`.
pub fn gpu_bitwise_hist_store(
    value: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = value.len();
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let v_d = stream.clone_htod(value)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_store)
                .arg(&n_u64)
                .arg(&v_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// BYTEWISE op-vec source: 8 BYTE_ALU per op (kind from `op`: 0/1/2 → AND/OR/XOR).
/// Bit-identical to `BytewiseOperation::collect_bitwise_ops`.
pub fn gpu_bitwise_hist_bytewise(
    a: &[u64],
    b: &[u64],
    op: &[u8],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), op.len());
    let n = a.len();
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let a_d = stream.clone_htod(a)?;
        let b_d = stream.clone_htod(b)?;
        let op_d = stream.clone_htod(op)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_bytewise)
                .arg(&n_u64)
                .arg(&a_d)
                .arg(&b_d)
                .arg(&op_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// EQ op-vec source: per `(a,b)` op, 4 IS_HALF over `(a-b)` halves + 1 ZERO[Σ halves].
/// Bit-identical to `EqOperation::collect_bitwise_ops`.
pub fn gpu_bitwise_hist_eq(
    a: &[u64],
    b: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let a_d = stream.clone_htod(a)?;
        let b_d = stream.clone_htod(b)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_eq)
                .arg(&n_u64)
                .arg(&a_d)
                .arg(&b_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// LOAD op-vec source: 1 Msb8 per op when width∈{1,2,4} (skip width 8). `res` is
/// 8 u64 per op (byte value per limb). Bit-identical to `LoadOperation::collect_bitwise_ops`.
pub fn gpu_bitwise_hist_load(
    res: &[u64],
    width: &[u32],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = width.len();
    assert_eq!(res.len(), n * 8);
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let res_d = stream.clone_htod(res)?;
        let w_d = stream.clone_htod(width)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_load)
                .arg(&n_u64)
                .arg(&res_d)
                .arg(&w_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// CPU32 op-vec source: per op, 5 ARE_BYTES + 8 IS_HALF + 1 BYTE_ALU[AND,32,alu_flags]
/// + (signed? 2 Msb16) + 1 Msb16. `signed` is derived on device from `alu_flags` bit 5.
/// Bit-identical to `collect_cpu32_bitwise`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_bitwise_hist_cpu32(
    hil: &[u8],
    alu_flags: &[u8],
    rs1: &[u8],
    rs2: &[u8],
    rd: &[u8],
    rv1: &[u64],
    rv2: &[u64],
    res: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = hil.len();
    assert!(
        [alu_flags.len(), rs1.len(), rs2.len(), rd.len(), rv1.len(), rv2.len(), res.len()]
            .iter()
            .all(|&l| l == n)
    );
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let hil_d = stream.clone_htod(hil)?;
        let af_d = stream.clone_htod(alu_flags)?;
        let rs1_d = stream.clone_htod(rs1)?;
        let rs2_d = stream.clone_htod(rs2)?;
        let rd_d = stream.clone_htod(rd)?;
        let rv1_d = stream.clone_htod(rv1)?;
        let rv2_d = stream.clone_htod(rv2)?;
        let res_d = stream.clone_htod(res)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_cpu32)
                .arg(&n_u64)
                .arg(&hil_d)
                .arg(&af_d)
                .arg(&rs1_d)
                .arg(&rs2_d)
                .arg(&rd_d)
                .arg(&rv1_d)
                .arg(&rv2_d)
                .arg(&res_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// BRANCH op-vec source: per op, ARE_BYTES + BYTE_ALU[AND] + 3 IS_HALF from the
/// (precomputed) `next_pc`/`next_pc_unmasked`. Bit-identical to `collect_bitwise_from_branch`.
pub fn gpu_bitwise_hist_branch(
    next_pc: &[u64],
    next_pc_unmasked: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(next_pc.len(), next_pc_unmasked.len());
    let n = next_pc.len();
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let npc_d = stream.clone_htod(next_pc)?;
        let unm_d = stream.clone_htod(next_pc_unmasked)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_branch)
                .arg(&n_u64)
                .arg(&npc_d)
                .arg(&unm_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// BRANCH + LOAD op-vec sources computed on-device from the resident cpu_ops seam
/// (`bitwise_hist_branch_load_packed`), host-in/host-out form for the parity test. Self-routes
/// by `packed`/`flags`; returns the `[num_rows*num_types]` counter array. Bit-identical to the SUM
/// of `gpu_bitwise_hist_branch` (fed the host `next_pc`/`unmasked`) and `gpu_bitwise_hist_load`
/// (fed the host load `res`/`width`) over the same cycles.
#[allow(clippy::too_many_arguments)]
pub fn gpu_bitwise_hist_branch_load_packed(
    packed: &[u64],
    flags: &[u8],
    pc: &[u64],
    imm: &[u64],
    rv1: &[u64],
    rvd: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = packed.len();
    assert!([flags.len(), pc.len(), imm.len(), rv1.len(), rvd.len()].iter().all(|&l| l == n));
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let pk_d = stream.clone_htod(packed)?;
        let fl_d = stream.clone_htod(flags)?;
        let pc_d = stream.clone_htod(pc)?;
        let imm_d = stream.clone_htod(imm)?;
        let rv1_d = stream.clone_htod(rv1)?;
        let rvd_d = stream.clone_htod(rvd)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_branch_load_packed)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&fl_d)
                .arg(&pc_d)
                .arg(&imm_d)
                .arg(&rv1_d)
                .arg(&rvd_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// CPU32 op-vec source from the PACKED device op rows (`bitwise_hist_cpu32_packed`), host-in/host-out
/// for the parity test. `rows` = `n_ops * 8` u64 (pack_cpu32_op layout). Bit-identical to
/// `gpu_bitwise_hist_cpu32` fed the same op's SoA (both == CPU `collect_cpu32_bitwise`).
pub fn gpu_bitwise_hist_cpu32_packed(
    rows: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(rows.len() % 8, 0, "cpu32 packed rows must be 8 u64/op");
    let n_ops = rows.len() / 8;
    run_source(num_rows, num_types, n_ops, |be, stream, nr, copies, stride, hist| {
        let rows_d = stream.clone_htod(rows)?;
        let n_u64 = n_ops as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_cpu32_packed)
                .arg(&n_u64)
                .arg(&rows_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n_ops as u32))?;
        }
        Ok(())
    })
}

/// SHIFT op-vec source from the PACKED device op rows (`bitwise_hist_shift_packed`), host-in/host-out
/// for the parity test. `rows` = `n_ops * 3` u64 ([value, shift_amount, flags], as `build_shift_ops`/
/// `cpu32_shift_ops` emit). Bit-identical to `gpu_bitwise_hist_shift` / CPU `collect_bitwise_from_shift`.
pub fn gpu_bitwise_hist_shift_packed(
    rows: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    assert_eq!(rows.len() % 3, 0, "shift packed rows must be 3 u64/op");
    let n_ops = rows.len() / 3;
    run_source(num_rows, num_types, n_ops, |be, stream, nr, copies, stride, hist| {
        let rows_d = stream.clone_htod(rows)?;
        let n_u64 = n_ops as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_shift_packed)
                .arg(&n_u64)
                .arg(&rows_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n_ops as u32))?;
        }
        Ok(())
    })
}

/// SHIFT op-vec source: per op (μ=1), recompute compute_aux on device then emit the
/// `collect_bitwise_from_shift` decomposition. `value` packs the 4 input halves; `flags`
/// bit0=direction(right), bit1=signed, bit2=word_instr. Bit-identical to `collect_bitwise_from_shift`.
pub fn gpu_bitwise_hist_shift(
    value: &[u64],
    shift: &[u8],
    shift_amount: &[u64],
    flags: &[u32],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = value.len();
    assert!([shift.len(), shift_amount.len(), flags.len()].iter().all(|&l| l == n));
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let v_d = stream.clone_htod(value)?;
        let s_d = stream.clone_htod(shift)?;
        let sa_d = stream.clone_htod(shift_amount)?;
        let fl_d = stream.clone_htod(flags)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_shift)
                .arg(&n_u64)
                .arg(&v_d)
                .arg(&s_d)
                .arg(&sa_d)
                .arg(&fl_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// MUL op-vec source (PER-OP part only): 16 IS_HALF + 4 IS_B20 per raw op. `flags` bit0=lhs_signed,
/// bit1=rhs_signed. The chunk-deduped MSB16 (signed sign bits) is emitted separately in P4b from the
/// deduped table rows; this matches `collect_bitwise_from_mul` exactly for UNSIGNED ops.
pub fn gpu_bitwise_hist_mul_perop(
    lhs: &[u64],
    rhs: &[u64],
    flags: &[u32],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = lhs.len();
    assert!([rhs.len(), flags.len()].iter().all(|&l| l == n));
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let lhs_d = stream.clone_htod(lhs)?;
        let rhs_d = stream.clone_htod(rhs)?;
        let fl_d = stream.clone_htod(flags)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_mul_perop)
                .arg(&n_u64)
                .arg(&lhs_d)
                .arg(&rhs_d)
                .arg(&fl_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// MUL per-op source from the MERGED device key stream (`bitwise_hist_mul_perop_packed`): `k0`=flags
/// (lhs_signed|rhs_signed<<1), `k1`=lhs, `k2`=rhs — the pre-dedup stream `mul_full_resident_core`
/// builds. Host-in/host-out for parity. Bit-identical to `gpu_bitwise_hist_mul_perop` / CPU
/// `collect_bitwise_from_mul` (per-op part).
pub fn gpu_bitwise_hist_mul_perop_packed(
    k0: &[u64],
    k1: &[u64],
    k2: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n_ops = k0.len();
    assert!([k1.len(), k2.len()].iter().all(|&l| l == n_ops));
    run_source(num_rows, num_types, n_ops, |be, stream, nr, copies, stride, hist| {
        let k0_d = stream.clone_htod(k0)?;
        let k1_d = stream.clone_htod(k1)?;
        let k2_d = stream.clone_htod(k2)?;
        let n_u64 = n_ops as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_mul_perop_packed)
                .arg(&n_u64)
                .arg(&k0_d)
                .arg(&k1_d)
                .arg(&k2_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n_ops as u32))?;
        }
        Ok(())
    })
}

/// DVRM op-vec source (PER-OP part only): 20 IS_HALF + 2 ZERO per raw op. `flags` bit0=signed. The
/// chunk-deduped MSB16 + NEG-template ZERO (signed only) ride on the deduped DVRM rows in P4b; this
/// matches `collect_bitwise_from_dvrm` exactly for UNSIGNED ops.
pub fn gpu_bitwise_hist_dvrm_perop(
    n_vals: &[u64],
    d_vals: &[u64],
    flags: &[u32],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n = n_vals.len();
    assert!([d_vals.len(), flags.len()].iter().all(|&l| l == n));
    run_source(num_rows, num_types, n, |be, stream, nr, copies, stride, hist| {
        let n_d = stream.clone_htod(n_vals)?;
        let d_d = stream.clone_htod(d_vals)?;
        let fl_d = stream.clone_htod(flags)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_dvrm_perop)
                .arg(&n_u64)
                .arg(&n_d)
                .arg(&d_d)
                .arg(&fl_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    })
}

/// DVRM per-op source from the MERGED device key stream (`bitwise_hist_dvrm_perop_packed`): `k0`=flags
/// (signed), `k1`=n, `k2`=d — the pre-dedup stream `dvrm_full_resident_core` builds. Host-in/host-out
/// for parity. Bit-identical to `gpu_bitwise_hist_dvrm_perop` / CPU `collect_bitwise_from_dvrm` (per-op).
pub fn gpu_bitwise_hist_dvrm_perop_packed(
    k0: &[u64],
    k1: &[u64],
    k2: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let n_ops = k0.len();
    assert!([k1.len(), k2.len()].iter().all(|&l| l == n_ops));
    run_source(num_rows, num_types, n_ops, |be, stream, nr, copies, stride, hist| {
        let k0_d = stream.clone_htod(k0)?;
        let k1_d = stream.clone_htod(k1)?;
        let k2_d = stream.clone_htod(k2)?;
        let n_u64 = n_ops as u64;
        unsafe {
            stream
                .launch_builder(&be.bitwise_hist_dvrm_perop_packed)
                .arg(&n_u64)
                .arg(&k0_d)
                .arg(&k1_d)
                .arg(&k2_d)
                .arg(&nr)
                .arg(&copies)
                .arg(&stride)
                .arg(hist)
                .launch(LaunchConfig::for_num_elems(n_ops as u32))?;
        }
        Ok(())
    })
}

/// RESIDENT in-walk BITWISE histogram: scatters the per-CPU-op range checks straight from the
/// device-resident `packed`+`res` buffers (the `DeviceCpuOpsResident` seam) — NO host SoA rebuild
/// or upload, the overhead that made the earlier partial GPU histogram lose. Unpacks the decode
/// fields on device (`bitwise_hist_cpu_ops_packed`), reduces the `HIST_COPIES` replicated copies,
/// and returns the `[num_rows*num_types]` counter array. Bit-identical to the in-walk contribution
/// of [`gpu_bitwise_hist`] (memw empty). `n` = cpu_op count. Empty input → zeros.
pub fn gpu_bitwise_hist_in_walk_devbuf(
    packed: &CudaSlice<u64>,
    res: &CudaSlice<u64>,
    n: usize,
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    if n == 0 {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let nr = num_rows as u64;
    let stride = total as u64;
    let n_u64 = n as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_cpu_ops_packed)
            .arg(&n_u64)
            .arg(packed)
            .arg(res)
            .arg(&nr)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(&mut hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
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

/// SoA inputs for the fully-covered op-vec histogram sources (lt, store, bytewise, eq, load, cpu32,
/// branch, shift). Each source's arrays are exactly what its per-op kernel consumes (see the
/// individual `gpu_bitwise_hist_*` wrappers). Empty slices skip that source.
#[derive(Default)]
pub struct OpVecSources<'a> {
    pub lt_lhs: &'a [u64],
    pub lt_rhs: &'a [u64],
    pub store_val: &'a [u64],
    pub bytewise_a: &'a [u64],
    pub bytewise_b: &'a [u64],
    pub bytewise_op: &'a [u8],
    pub eq_a: &'a [u64],
    pub eq_b: &'a [u64],
    pub load_res: &'a [u64], // 8 per op
    pub load_width: &'a [u32],
    pub cpu32_hil: &'a [u8],
    pub cpu32_alu: &'a [u8],
    pub cpu32_rs1: &'a [u8],
    pub cpu32_rs2: &'a [u8],
    pub cpu32_rd: &'a [u8],
    pub cpu32_rv1: &'a [u64],
    pub cpu32_rv2: &'a [u64],
    pub cpu32_res: &'a [u64],
    pub branch_next_pc: &'a [u64],
    pub branch_unmasked: &'a [u64],
    pub shift_value: &'a [u64],
    pub shift_shift: &'a [u8],
    pub shift_amount: &'a [u64],
    pub shift_flags: &'a [u32],
    /// MEMW_ALIGNED source: one IS_HALF[base_low + mask(width)] per aligned memw op.
    pub memw_aligned_base: &'a [u64],
    pub memw_aligned_width: &'a [u32],
    pub memw_aligned_flag: &'a [u32], // 1 per aligned op (the aligned bucket is all-aligned)
    /// MUL/DVRM PER-OP part (IS_HALF/IS_B20 for mul; IS_HALF/ZERO for dvrm). The chunk-deduped MSB16
    /// (+ dvrm NEG-ZERO) tail stays on CPU. `mul_flags` bit0=lhs_signed/bit1=rhs_signed; `dvrm_flags` bit0=signed.
    pub mul_lhs: &'a [u64],
    pub mul_rhs: &'a [u64],
    pub mul_flags: &'a [u32],
    pub dvrm_n: &'a [u64],
    pub dvrm_d: &'a [u64],
    pub dvrm_flags: &'a [u32],
}

/// Scatter every op-vec source in `src` into the provided replicated histogram `hist` (no alloc, no
/// reduce) — the shared core used by both `gpu_bitwise_hist_opvec` (standalone) and
/// `gpu_bitwise_hist_full` (FR3: one histogram for ALL sources). Empty slices skip that source.
#[allow(clippy::too_many_arguments)]
fn scatter_opvec(
    be: &Backend,
    stream: &std::sync::Arc<CudaStream>,
    src: &OpVecSources,
    nr: u64,
    stride: u64,
    hist: &mut CudaSlice<u64>,
) -> Result<()> {
    // LT: 2 Msb16 + 6 IS_HALF per op.
    if !src.lt_lhs.is_empty() {
        let lhs = stream.clone_htod(src.lt_lhs)?;
        let rhs = stream.clone_htod(src.lt_rhs)?;
        let n = src.lt_lhs.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_lt).arg(&n).arg(&lhs).arg(&rhs).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.lt_lhs.len() as u32))?;
        }
    }
    // STORE: 8 ARE_BYTES per op.
    if !src.store_val.is_empty() {
        let v = stream.clone_htod(src.store_val)?;
        let n = src.store_val.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_store).arg(&n).arg(&v).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.store_val.len() as u32))?;
        }
    }
    // BYTEWISE: 8 BYTE_ALU per op.
    if !src.bytewise_a.is_empty() {
        let a = stream.clone_htod(src.bytewise_a)?;
        let b = stream.clone_htod(src.bytewise_b)?;
        let op = stream.clone_htod(src.bytewise_op)?;
        let n = src.bytewise_a.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_bytewise).arg(&n).arg(&a).arg(&b).arg(&op).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.bytewise_a.len() as u32))?;
        }
    }
    // EQ: 4 IS_HALF + 1 ZERO per op.
    if !src.eq_a.is_empty() {
        let a = stream.clone_htod(src.eq_a)?;
        let b = stream.clone_htod(src.eq_b)?;
        let n = src.eq_a.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_eq).arg(&n).arg(&a).arg(&b).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.eq_a.len() as u32))?;
        }
    }
    // LOAD: 1 Msb8 per op (width != 8).
    if !src.load_width.is_empty() {
        let res = stream.clone_htod(src.load_res)?;
        let w = stream.clone_htod(src.load_width)?;
        let n = src.load_width.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_load).arg(&n).arg(&res).arg(&w).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.load_width.len() as u32))?;
        }
    }
    // CPU32.
    if !src.cpu32_hil.is_empty() {
        let hil = stream.clone_htod(src.cpu32_hil)?;
        let alu = stream.clone_htod(src.cpu32_alu)?;
        let rs1 = stream.clone_htod(src.cpu32_rs1)?;
        let rs2 = stream.clone_htod(src.cpu32_rs2)?;
        let rd = stream.clone_htod(src.cpu32_rd)?;
        let rv1 = stream.clone_htod(src.cpu32_rv1)?;
        let rv2 = stream.clone_htod(src.cpu32_rv2)?;
        let res = stream.clone_htod(src.cpu32_res)?;
        let n = src.cpu32_hil.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_cpu32).arg(&n).arg(&hil).arg(&alu).arg(&rs1)
                .arg(&rs2).arg(&rd).arg(&rv1).arg(&rv2).arg(&res).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.cpu32_hil.len() as u32))?;
        }
    }
    // BRANCH.
    if !src.branch_next_pc.is_empty() {
        let npc = stream.clone_htod(src.branch_next_pc)?;
        let unm = stream.clone_htod(src.branch_unmasked)?;
        let n = src.branch_next_pc.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_branch).arg(&n).arg(&npc).arg(&unm).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.branch_next_pc.len() as u32))?;
        }
    }
    // SHIFT.
    if !src.shift_value.is_empty() {
        let v = stream.clone_htod(src.shift_value)?;
        let s = stream.clone_htod(src.shift_shift)?;
        let sa = stream.clone_htod(src.shift_amount)?;
        let fl = stream.clone_htod(src.shift_flags)?;
        let n = src.shift_value.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_shift).arg(&n).arg(&v).arg(&s).arg(&sa).arg(&fl)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.shift_value.len() as u32))?;
        }
    }
    // MEMW_ALIGNED: 1 IS_HALF[base_low + mask(width)] per aligned op.
    if !src.memw_aligned_base.is_empty() {
        let base = stream.clone_htod(src.memw_aligned_base)?;
        let width = stream.clone_htod(src.memw_aligned_width)?;
        let flag = stream.clone_htod(src.memw_aligned_flag)?;
        let n = src.memw_aligned_base.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_memw_aligned).arg(&n).arg(&base).arg(&width)
                .arg(&flag).arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.memw_aligned_base.len() as u32))?;
        }
    }
    // MUL per-op: 16 IS_HALF + 4 IS_B20.
    if !src.mul_lhs.is_empty() {
        let lhs = stream.clone_htod(src.mul_lhs)?;
        let rhs = stream.clone_htod(src.mul_rhs)?;
        let fl = stream.clone_htod(src.mul_flags)?;
        let n = src.mul_lhs.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_mul_perop).arg(&n).arg(&lhs).arg(&rhs).arg(&fl)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.mul_lhs.len() as u32))?;
        }
    }
    // DVRM per-op: 20 IS_HALF + 2 ZERO.
    if !src.dvrm_n.is_empty() {
        let nn = stream.clone_htod(src.dvrm_n)?;
        let dd = stream.clone_htod(src.dvrm_d)?;
        let fl = stream.clone_htod(src.dvrm_flags)?;
        let n = src.dvrm_n.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_dvrm_perop).arg(&n).arg(&nn).arg(&dd).arg(&fl)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut *hist)
                .launch(LaunchConfig::for_num_elems(src.dvrm_n.len() as u32))?;
        }
    }
    Ok(())
}

/// FR3 — the WHOLE resident BITWISE histogram in ONE replicated buffer with ONE reduce/download:
/// in_walk (resident `packed`/`res`) + op-vec (`src`) + page (`init`/`fini`) + memw_reg (resident
/// register walk, ecall-interleaved, routed). Returns the `[num_rows*num_types]` counter array,
/// bit-identical to the sum of `gpu_bitwise_hist_resident_devops` (in_walk+page) + `gpu_bitwise_hist_opvec`
/// + `gpu_memw_reg_hist_resident_ecall`. The only host merge left for the caller is the tiny EC/precompile
/// CPU residual. `packed_dev` is reused for both in_walk and the memw_reg walk emit (no re-upload).
#[allow(clippy::too_many_arguments)]
pub fn gpu_bitwise_hist_full(
    packed_dev: &CudaSlice<u64>,
    res_dev: &CudaSlice<u64>,
    n: usize,
    rv1_dev: &CudaSlice<u64>,
    rv2_dev: &CudaSlice<u64>,
    arg2_dev: &CudaSlice<u64>,
    rvd_dev: &CudaSlice<u64>,
    pc_dev: &CudaSlice<u64>,
    imm_dev: &CudaSlice<u64>,
    flags_dev: &CudaSlice<u8>,
    next_pc: &[u64],
    ecall_op_index: &[u32],
    ecall_reg_addr: &[u32],
    ecall_ts: &[u64],
    ecall_value: &[u64],
    ecall_is_read: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    page_init: &[u8],
    page_fini: &[u8],
    src: &OpVecSources,
    num_rows: usize,
    num_types: usize,
    // Fix (a) walk-once: when the MEMW_R IS_HALF histogram was already computed by the register-table
    // walk, SKIP the redundant second register walk here (caller merges those counts via `device_is_half`).
    skip_memw_reg: bool,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let nr = num_rows as u64;
    let stride = total as u64;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;

    // in_walk (resident packed/res).
    if n > 0 {
        let n_u64 = n as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_cpu_ops_packed).arg(&n_u64).arg(packed_dev)
                .arg(res_dev).arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // FR4b: resident op-vec (STORE + EQ + BYTEWISE) computed on-device from the resident
    // packed/rv1/rv2/arg2 (no host SoA). The caller leaves these three empty in `src`, so
    // `scatter_opvec` skips them. S3 adds BRANCH + LOAD + CPU32 + SHIFT + MUL + DVRM (per-op) + the
    // instruction/dvrm→LT (STEP 2A, device key gathers) below; `scatter_opvec` still handles the
    // memw→lt pairs (`src.lt_*`, device-DERIVED but fed as arrays) + memw_aligned.
    if n > 0 {
        let n_u64 = n as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_opvec_packed).arg(&n_u64).arg(packed_dev)
                .arg(rv1_dev).arg(rv2_dev).arg(arg2_dev).arg(&nr).arg(&HIST_COPIES).arg(&stride)
                .arg(&mut hist).launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // S3: resident op-vec (BRANCH + LOAD) computed on-device from the resident packed/flags/pc/imm/
    // rv1/rvd seam (no host SoA). The caller leaves `branch_*`/`load_*` empty in `src`, so
    // `scatter_opvec` skips them.
    if n > 0 {
        let n_u64 = n as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_branch_load_packed).arg(&n_u64).arg(packed_dev)
                .arg(flags_dev).arg(pc_dev).arg(imm_dev).arg(rv1_dev).arg(rvd_dev).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    // S3: resident CPU32 op-vec — reuse the device CPU32 op-build (`build_cpu32_ops`; res computed on
    // device, validated == build_cpu32_op) then scatter `collect_cpu32_bitwise` from the packed rows
    // (`bitwise_hist_cpu32_packed`). No host SoA. The caller leaves `cpu32_*` empty in `src`.
    // NOTE: this adds route+scan+build passes over ALL n cycles to harvest ~3.9K cpu32 ops, so it is
    // ~40-70ms SLOWER warm — kept ON for the "whole trace-gen on GPU" completeness mandate (speed is
    // not the goal). A future shared cpu32 op-stream across p4(hist)+p5(table) would make it free.
    if n > 0 {
        let n_u64 = n as u64;
        let mut flag = stream.alloc_zeros::<u32>(n)?;
        unsafe {
            stream.launch_builder(&be.cpu32_route).arg(&n_u64).arg(packed_dev).arg(&mut flag)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        let (excl, total) = crate::trace_walk::excl_scan(be, &stream, &flag, n)?;
        let rows = total as usize;
        if rows > 0 {
            let mut ops_dev = stream.alloc_zeros::<u64>(rows * 8)?;
            unsafe {
                stream.launch_builder(&be.build_cpu32_ops).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                    .arg(rv2_dev).arg(imm_dev).arg(pc_dev).arg(&flag).arg(&excl).arg(&mut ops_dev)
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
            let rows_u64 = rows as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_cpu32_packed).arg(&rows_u64).arg(&ops_dev)
                    .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                    .launch(LaunchConfig::for_num_elems(rows as u32))?;
            }
        }
    }
    // S3: resident SHIFT op-vec — reuse the device SHIFT op-build (instruction ⊕ cpu32-derived,
    // `build_shift_ops`+`cpu32_shift_ops` → 3-u64 rows [value, shift_amount, flags]) then scatter
    // `collect_bitwise_from_shift` via `bitwise_hist_shift_packed`. No host SoA. The caller leaves
    // `shift_*` empty in `src`. (Same full-n-pass cost note as cpu32; kept ON for completeness.)
    if n > 0 {
        let n_u64 = n as u64;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut f0 = stream.alloc_zeros::<u32>(n)?;
        let mut f1 = stream.alloc_zeros::<u32>(n)?; // alu[1] = instruction SHIFT
        let mut f2 = stream.alloc_zeros::<u32>(n)?;
        let mut f3 = stream.alloc_zeros::<u32>(n)?;
        let mut f4 = stream.alloc_zeros::<u32>(n)?;
        let mut f5 = stream.alloc_zeros::<u32>(n)?;
        let mut cpu32_shift = stream.alloc_zeros::<u32>(n)?;
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(packed_dev)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5)
                .launch(cfg)?;
            stream.launch_builder(&be.cpu32_shift_route).arg(&n_u64).arg(packed_dev)
                .arg(&mut cpu32_shift).launch(cfg)?;
        }
        let (excl_i, total_i) = crate::trace_walk::excl_scan(be, &stream, &f1, n)?;
        let (excl_c, total_c) = crate::trace_walk::excl_scan(be, &stream, &cpu32_shift, n)?;
        let rows_i = total_i as usize;
        let rows_c = total_c as usize;
        let total = rows_i + rows_c;
        if total > 0 {
            let mut ops_dev = stream.alloc_zeros::<u64>(total * 3)?;
            if rows_i > 0 {
                unsafe {
                    stream.launch_builder(&be.build_shift_ops).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(arg2_dev).arg(&f1).arg(&excl_i).arg(&mut ops_dev).launch(cfg)?;
                }
            }
            if rows_c > 0 {
                let base = rows_i as u64;
                unsafe {
                    stream.launch_builder(&be.cpu32_shift_ops).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(rv2_dev).arg(imm_dev).arg(&cpu32_shift).arg(&excl_c).arg(&base)
                        .arg(&mut ops_dev).launch(cfg)?;
                }
            }
            let total_u64 = total as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_shift_packed).arg(&total_u64).arg(&ops_dev)
                    .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                    .launch(LaunchConfig::for_num_elems(total as u32))?;
            }
        }
    }
    // S3: resident MUL per-op — reuse the device MUL key gather (4 sources merged pre-dedup:
    // instruction ⊕ instruction-dvrm→mul ⊕ cpu32 ⊕ cpu32-dvrm→mul) then scatter the per-op bitwise
    // (`bitwise_hist_mul_perop_packed`). The chunk-deduped MSB16 tail stays host (rides the deduped MUL
    // table). No host per-op SoA. Mirrors `mul_full_resident_core`'s gather.
    if n > 0 {
        let n_u64 = n as u64;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut f0 = stream.alloc_zeros::<u32>(n)?;
        let mut f1 = stream.alloc_zeros::<u32>(n)?;
        let mut f2 = stream.alloc_zeros::<u32>(n)?;
        let mut f3 = stream.alloc_zeros::<u32>(n)?;
        let mut f4 = stream.alloc_zeros::<u32>(n)?; // MUL
        let mut f5 = stream.alloc_zeros::<u32>(n)?; // DVRM
        let mut c_mul = stream.alloc_zeros::<u32>(n)?;
        let mut c_dvrm = stream.alloc_zeros::<u32>(n)?;
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(packed_dev)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5)
                .launch(cfg)?;
            stream.launch_builder(&be.cpu32_mul_route).arg(&n_u64).arg(packed_dev).arg(&mut c_mul).launch(cfg)?;
            stream.launch_builder(&be.cpu32_dvrm_route).arg(&n_u64).arg(packed_dev).arg(&mut c_dvrm).launch(cfg)?;
        }
        let (excl_m, tm) = crate::trace_walk::excl_scan(be, &stream, &f4, n)?;
        let (excl_d, td) = crate::trace_walk::excl_scan(be, &stream, &f5, n)?;
        let (excl_c, tc) = crate::trace_walk::excl_scan(be, &stream, &c_mul, n)?;
        let (excl_cd, tcd) = crate::trace_walk::excl_scan(be, &stream, &c_dvrm, n)?;
        let (rm, rd, rc, rcd) = (tm as usize, td as usize, tc as usize, tcd as usize);
        let total = rm + 2 * rd + rc + 2 * rcd;
        if total > 0 {
            let mut k0 = stream.alloc_zeros::<u64>(total)?;
            let mut k1 = stream.alloc_zeros::<u64>(total)?;
            let mut k2 = stream.alloc_zeros::<u64>(total)?;
            let mut sel = stream.alloc_zeros::<u32>(total)?;
            if rm > 0 {
                unsafe {
                    stream.launch_builder(&be.mul_key_gather).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(arg2_dev).arg(&f4).arg(&excl_m).arg(&mut k0).arg(&mut k1).arg(&mut k2)
                        .arg(&mut sel).launch(cfg)?;
                }
            }
            if rd > 0 {
                let base = rm as u64;
                unsafe {
                    stream.launch_builder(&be.mul_dvrm_key_gather).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(arg2_dev).arg(&f5).arg(&excl_d).arg(&base).arg(&mut k0).arg(&mut k1)
                        .arg(&mut k2).arg(&mut sel).launch(cfg)?;
                }
            }
            if rc > 0 {
                let base = (rm + 2 * rd) as u64;
                unsafe {
                    stream.launch_builder(&be.cpu32_mul_ops).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(rv2_dev).arg(imm_dev).arg(&c_mul).arg(&excl_c).arg(&base).arg(&mut k0)
                        .arg(&mut k1).arg(&mut k2).arg(&mut sel).launch(cfg)?;
                }
            }
            if rcd > 0 {
                let base = (rm + 2 * rd + rc) as u64;
                unsafe {
                    stream.launch_builder(&be.cpu32_dvrm_mul_key_gather).arg(&n_u64).arg(packed_dev)
                        .arg(rv1_dev).arg(rv2_dev).arg(imm_dev).arg(&c_dvrm).arg(&excl_cd).arg(&base)
                        .arg(&mut k0).arg(&mut k1).arg(&mut k2).arg(&mut sel).launch(cfg)?;
                }
            }
            let total_u64 = total as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_mul_perop_packed).arg(&total_u64).arg(&k0)
                    .arg(&k1).arg(&k2).arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                    .launch(LaunchConfig::for_num_elems(total as u32))?;
            }
        }
    }
    // S3: resident DVRM per-op — reuse the device DVRM key gather (instruction ⊕ cpu32-derived) then
    // scatter the per-op bitwise (`bitwise_hist_dvrm_perop_packed`). The chunk-deduped MSB16/NEG-ZERO
    // tail stays host. No host per-op SoA. Mirrors `dvrm_full_resident_core`'s gather.
    if n > 0 {
        let n_u64 = n as u64;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut f0 = stream.alloc_zeros::<u32>(n)?;
        let mut f1 = stream.alloc_zeros::<u32>(n)?;
        let mut f2 = stream.alloc_zeros::<u32>(n)?;
        let mut f3 = stream.alloc_zeros::<u32>(n)?;
        let mut f4 = stream.alloc_zeros::<u32>(n)?;
        let mut f5 = stream.alloc_zeros::<u32>(n)?; // DVRM
        let mut c_dvrm = stream.alloc_zeros::<u32>(n)?;
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(packed_dev)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5)
                .launch(cfg)?;
            stream.launch_builder(&be.cpu32_dvrm_route).arg(&n_u64).arg(packed_dev).arg(&mut c_dvrm).launch(cfg)?;
        }
        let (excl_i, ti) = crate::trace_walk::excl_scan(be, &stream, &f5, n)?;
        let (excl_c, tc) = crate::trace_walk::excl_scan(be, &stream, &c_dvrm, n)?;
        let (ri, rc) = (ti as usize, tc as usize);
        let total = ri + rc;
        if total > 0 {
            let mut k0 = stream.alloc_zeros::<u64>(total)?;
            let mut k1 = stream.alloc_zeros::<u64>(total)?;
            let mut k2 = stream.alloc_zeros::<u64>(total)?;
            let mut sel = stream.alloc_zeros::<u32>(total)?;
            if ri > 0 {
                unsafe {
                    stream.launch_builder(&be.dvrm_key_gather).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(arg2_dev).arg(&f5).arg(&excl_i).arg(&mut k0).arg(&mut k1).arg(&mut k2)
                        .arg(&mut sel).launch(cfg)?;
                }
            }
            if rc > 0 {
                let base = ri as u64;
                unsafe {
                    stream.launch_builder(&be.cpu32_dvrm_ops).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                        .arg(rv2_dev).arg(imm_dev).arg(&c_dvrm).arg(&excl_c).arg(&base).arg(&mut k0)
                        .arg(&mut k1).arg(&mut k2).arg(&mut sel).launch(cfg)?;
                }
            }
            let total_u64 = total as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_dvrm_perop_packed).arg(&total_u64).arg(&k0)
                    .arg(&k1).arg(&k2).arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                    .launch(LaunchConfig::for_num_elems(total as u32))?;
            }
        }
    }
    // S3/LT STEP 2A: instruction LT + dvrm→lt op-vec scattered ON DEVICE (`bitwise_hist_lt` from the
    // resident key gathers) — the memw→lt part rides in `src.lt_*` as device-derived pairs. LT bitwise
    // depends only on (lhs, rhs), so scatter the compacted key streams directly (k1=lhs, k2=rhs).
    if n > 0 {
        let n_u64 = n as u64;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut f0 = stream.alloc_zeros::<u32>(n)?; // LT
        let mut f1 = stream.alloc_zeros::<u32>(n)?;
        let mut f2 = stream.alloc_zeros::<u32>(n)?;
        let mut f3 = stream.alloc_zeros::<u32>(n)?;
        let mut f4 = stream.alloc_zeros::<u32>(n)?;
        let mut f5 = stream.alloc_zeros::<u32>(n)?; // DVRM
        unsafe {
            stream.launch_builder(&be.chipop_alu_route).arg(&n_u64).arg(packed_dev)
                .arg(&mut f0).arg(&mut f1).arg(&mut f2).arg(&mut f3).arg(&mut f4).arg(&mut f5)
                .launch(cfg)?;
        }
        // instruction LT (f0)
        let (excl_lt, total_lt) = crate::trace_walk::excl_scan(be, &stream, &f0, n)?;
        let rows_lt = total_lt as usize;
        if rows_lt > 0 {
            let mut k0 = stream.alloc_zeros::<u64>(rows_lt)?;
            let mut k1 = stream.alloc_zeros::<u64>(rows_lt)?;
            let mut k2 = stream.alloc_zeros::<u64>(rows_lt)?;
            unsafe {
                stream.launch_builder(&be.lt_key_gather).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                    .arg(arg2_dev).arg(&f0).arg(&excl_lt).arg(&mut k0).arg(&mut k1).arg(&mut k2)
                    .launch(cfg)?;
            }
            let r = rows_lt as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_lt).arg(&r).arg(&k1).arg(&k2).arg(&nr)
                    .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                    .launch(LaunchConfig::for_num_elems(rows_lt as u32))?;
            }
        }
        // dvrm→lt (f5)
        let (excl_dv, total_dv) = crate::trace_walk::excl_scan(be, &stream, &f5, n)?;
        let rows_dv = total_dv as usize;
        if rows_dv > 0 {
            let mut k0 = stream.alloc_zeros::<u64>(rows_dv)?;
            let mut k1 = stream.alloc_zeros::<u64>(rows_dv)?;
            let mut k2 = stream.alloc_zeros::<u64>(rows_dv)?;
            let base = 0u64;
            unsafe {
                stream.launch_builder(&be.dvrm_lt_key_gather).arg(&n_u64).arg(packed_dev).arg(rv1_dev)
                    .arg(arg2_dev).arg(&f5).arg(&excl_dv).arg(&base).arg(&mut k0).arg(&mut k1).arg(&mut k2)
                    .launch(cfg)?;
            }
            let r = rows_dv as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_lt).arg(&r).arg(&k1).arg(&k2).arg(&nr)
                    .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                    .launch(LaunchConfig::for_num_elems(rows_dv as u32))?;
            }
        }
    }
    // op-vec (still-host sources).
    scatter_opvec(be, &stream, src, nr, stride, &mut hist)?;
    // page.
    if !page_init.is_empty() {
        let init_d = stream.clone_htod(page_init)?;
        let fini_d = stream.clone_htod(page_fini)?;
        let pn = page_init.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_page).arg(&pn).arg(&init_d).arg(&fini_d).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(page_init.len() as u32))?;
        }
    }
    // memw_reg — resident walk: reuse the resident packed/rv1/rv2/rvd seam (only next_pc uploaded) +
    // routed scatter into the shared hist. Fix (a): SKIPPED when the register-table walk already
    // produced these IS_HALF counts (caller merges them via `device_is_half`) — avoids re-walking
    // the ~13M register accesses a second time.
    if n > 0 && !skip_memw_reg {
        let npc_d = stream.clone_htod(next_pc)?;
        let (reg_addr_d, ts_d, value_d, _is_read_d, row_index_d, total_acc) =
            crate::trace_walk::emit_register_accesses_with_ecall_dev(
                be, &stream, packed_dev, rv1_dev, rv2_dev, rvd_dev, &npc_d, n, ecall_op_index,
                ecall_reg_addr, ecall_ts, ecall_value, ecall_is_read,
            )?;
        if total_acc > 0 {
            let init_value_d = stream.clone_htod(init_value)?;
            let (_old_value_d, old_ts_d) = crate::trace_walk::walk_core(
                be, &stream, &reg_addr_d, &ts_d, &value_d, &init_value_d, init_ts, total_acc, nbins,
            )?;
            let n_acc = total_acc as u64;
            unsafe {
                stream.launch_builder(&be.bitwise_hist_memw_reg_routed).arg(&n_acc).arg(&ts_d)
                    .arg(&old_ts_d).arg(&row_index_d).arg(&nr).arg(&HIST_COPIES).arg(&stride)
                    .arg(&mut hist).launch(LaunchConfig::for_num_elems(total_acc as u32))?;
            }
        }
    }

    stream.synchronize()?;
    unsafe {
        stream.launch_builder(&be.bitwise_hist_reduce).arg(&stride).arg(&HIST_COPIES).arg(&hist)
            .arg(&mut out).launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// FR4a — RESIDENT PAGE histogram: compute the PAGE ARE_BYTES[init, fini] multiplicities on-device
/// from the sorted initial image (`img_addr`/`img_val`) + the device final-memory snapshot
/// (`snap_addr`/`snap_val`), returning the `[num_rows*num_types]` counter array. Replaces the ~1s
/// host `build_page_bitwise_arrays` (HashMap over ~4.7M cells) — the device kernel binary-searches
/// image + snapshot per byte. Bit-identical to `collect_bitwise_from_page` (page_bases = pages of the
/// reconstructed state; a byte's fini = snapshot value if touched, else its init). `page_bases` must
/// be ascending page-aligned bases; `img_*` and `snap_*` ascending by address.
#[allow(clippy::too_many_arguments)]
pub fn gpu_bitwise_hist_page_snapshot(
    page_bases: &[u64],
    page_size: u64,
    img_addr: &[u64],
    img_val: &[u64],
    snap_addr: &[u64],
    snap_val: &[u64],
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let nr = num_rows as u64;
    let stride = total as u64;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;

    let n_bytes = page_bases.len() as u64 * page_size;
    if n_bytes > 0 {
        let pb = stream.clone_htod(page_bases)?;
        let ia = stream.clone_htod(img_addr)?;
        let iv = stream.clone_htod(img_val)?;
        let sa = stream.clone_htod(snap_addr)?;
        let sv = stream.clone_htod(snap_val)?;
        let num_pages = page_bases.len() as u64;
        let img_n = img_addr.len() as u64;
        let snap_n = snap_addr.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_page_snapshot).arg(&pb).arg(&num_pages)
                .arg(&page_size).arg(&ia).arg(&iv).arg(&img_n).arg(&sa).arg(&sv).arg(&snap_n)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(n_bytes as u32))?;
        }
    }
    unsafe {
        stream.launch_builder(&be.bitwise_hist_reduce).arg(&stride).arg(&HIST_COPIES).arg(&hist)
            .arg(&mut out).launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// P4b assembly: scatter ALL fully-covered op-vec sources into ONE replicated histogram and reduce
/// once, returning the `[num_rows*num_types]` counter array. Bit-identical to the sum of the
/// per-source CPU collectors (each source's kernel is already validated 1:1). This is the op-vec
/// portion of the resident BITWISE histogram; the walk/page/in_walk sources are added separately.
pub fn gpu_bitwise_hist_opvec(
    src: &OpVecSources,
    num_rows: usize,
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let total = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total)?;
    let nr = num_rows as u64;
    let stride = total as u64;
    let mut hist = stream.alloc_zeros::<u64>(total * HIST_COPIES as usize)?;

    // LT: 2 Msb16 + 6 IS_HALF per op.
    if !src.lt_lhs.is_empty() {
        let lhs = stream.clone_htod(src.lt_lhs)?;
        let rhs = stream.clone_htod(src.lt_rhs)?;
        let n = src.lt_lhs.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_lt).arg(&n).arg(&lhs).arg(&rhs).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.lt_lhs.len() as u32))?;
        }
    }
    // STORE: 8 ARE_BYTES per op.
    if !src.store_val.is_empty() {
        let v = stream.clone_htod(src.store_val)?;
        let n = src.store_val.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_store).arg(&n).arg(&v).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.store_val.len() as u32))?;
        }
    }
    // BYTEWISE: 8 BYTE_ALU per op.
    if !src.bytewise_a.is_empty() {
        let a = stream.clone_htod(src.bytewise_a)?;
        let b = stream.clone_htod(src.bytewise_b)?;
        let op = stream.clone_htod(src.bytewise_op)?;
        let n = src.bytewise_a.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_bytewise).arg(&n).arg(&a).arg(&b).arg(&op).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.bytewise_a.len() as u32))?;
        }
    }
    // EQ: 4 IS_HALF + 1 ZERO per op.
    if !src.eq_a.is_empty() {
        let a = stream.clone_htod(src.eq_a)?;
        let b = stream.clone_htod(src.eq_b)?;
        let n = src.eq_a.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_eq).arg(&n).arg(&a).arg(&b).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.eq_a.len() as u32))?;
        }
    }
    // LOAD: 1 Msb8 per op (width != 8).
    if !src.load_width.is_empty() {
        let res = stream.clone_htod(src.load_res)?;
        let w = stream.clone_htod(src.load_width)?;
        let n = src.load_width.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_load).arg(&n).arg(&res).arg(&w).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.load_width.len() as u32))?;
        }
    }
    // CPU32.
    if !src.cpu32_hil.is_empty() {
        let hil = stream.clone_htod(src.cpu32_hil)?;
        let alu = stream.clone_htod(src.cpu32_alu)?;
        let rs1 = stream.clone_htod(src.cpu32_rs1)?;
        let rs2 = stream.clone_htod(src.cpu32_rs2)?;
        let rd = stream.clone_htod(src.cpu32_rd)?;
        let rv1 = stream.clone_htod(src.cpu32_rv1)?;
        let rv2 = stream.clone_htod(src.cpu32_rv2)?;
        let res = stream.clone_htod(src.cpu32_res)?;
        let n = src.cpu32_hil.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_cpu32).arg(&n).arg(&hil).arg(&alu).arg(&rs1)
                .arg(&rs2).arg(&rd).arg(&rv1).arg(&rv2).arg(&res).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.cpu32_hil.len() as u32))?;
        }
    }
    // BRANCH.
    if !src.branch_next_pc.is_empty() {
        let npc = stream.clone_htod(src.branch_next_pc)?;
        let unm = stream.clone_htod(src.branch_unmasked)?;
        let n = src.branch_next_pc.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_branch).arg(&n).arg(&npc).arg(&unm).arg(&nr)
                .arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.branch_next_pc.len() as u32))?;
        }
    }
    // SHIFT.
    if !src.shift_value.is_empty() {
        let v = stream.clone_htod(src.shift_value)?;
        let s = stream.clone_htod(src.shift_shift)?;
        let sa = stream.clone_htod(src.shift_amount)?;
        let fl = stream.clone_htod(src.shift_flags)?;
        let n = src.shift_value.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_shift).arg(&n).arg(&v).arg(&s).arg(&sa).arg(&fl)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.shift_value.len() as u32))?;
        }
    }
    // MEMW_ALIGNED: 1 IS_HALF[base_low + mask(width)] per aligned op.
    if !src.memw_aligned_base.is_empty() {
        let base = stream.clone_htod(src.memw_aligned_base)?;
        let width = stream.clone_htod(src.memw_aligned_width)?;
        let flag = stream.clone_htod(src.memw_aligned_flag)?;
        let n = src.memw_aligned_base.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_memw_aligned).arg(&n).arg(&base).arg(&width)
                .arg(&flag).arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.memw_aligned_base.len() as u32))?;
        }
    }
    // MUL per-op: 16 IS_HALF + 4 IS_B20.
    if !src.mul_lhs.is_empty() {
        let lhs = stream.clone_htod(src.mul_lhs)?;
        let rhs = stream.clone_htod(src.mul_rhs)?;
        let fl = stream.clone_htod(src.mul_flags)?;
        let n = src.mul_lhs.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_mul_perop).arg(&n).arg(&lhs).arg(&rhs).arg(&fl)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.mul_lhs.len() as u32))?;
        }
    }
    // DVRM per-op: 20 IS_HALF + 2 ZERO.
    if !src.dvrm_n.is_empty() {
        let nn = stream.clone_htod(src.dvrm_n)?;
        let dd = stream.clone_htod(src.dvrm_d)?;
        let fl = stream.clone_htod(src.dvrm_flags)?;
        let n = src.dvrm_n.len() as u64;
        unsafe {
            stream.launch_builder(&be.bitwise_hist_dvrm_perop).arg(&n).arg(&nn).arg(&dd).arg(&fl)
                .arg(&nr).arg(&HIST_COPIES).arg(&stride).arg(&mut hist)
                .launch(LaunchConfig::for_num_elems(src.dvrm_n.len() as u32))?;
        }
    }

    unsafe {
        stream.launch_builder(&be.bitwise_hist_reduce).arg(&stride).arg(&HIST_COPIES).arg(&hist)
            .arg(&mut out).launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// P4b/residency: the memw_reg BITWISE source computed from the RESIDENT register walk (with ecall
/// accesses interleaved), NO upload of the walk rows. Chains
/// `emit_register_accesses_with_ecall_dev` → `walk_core` → `bitwise_hist_memw_reg_routed` (which counts
/// only rows that stay in MEMW_R, excluding the reg_ts_delta fallback that routes to MEMW_A/MEMW on the
/// host) → reduce. Bit-identical to the host `collect_bitwise_from_memw_register` over `memw_register_rows`.
/// Only the cpu_op SoA + tiny ecall arrays + `init_value[nbins]` are uploaded (the walk rows stay resident).
#[allow(clippy::too_many_arguments)]
pub fn gpu_memw_reg_hist_resident_ecall(
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
    num_types: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let total_ct = num_rows * num_types;
    let mut out = stream.alloc_zeros::<u64>(total_ct)?;
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;
    let (reg_addr_d, ts_d, _value_d, _is_read_d, row_index_d, total) =
        crate::trace_walk::emit_register_accesses_with_ecall_dev(
            be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n, ecall_op_index, ecall_reg_addr,
            ecall_ts, ecall_value, ecall_is_read,
        )?;
    if total == 0 {
        let host = stream.clone_dtoh(&out)?;
        stream.synchronize()?;
        return Ok(host);
    }
    let init_value_d = stream.clone_htod(init_value)?;
    let (_old_value_d, old_ts_d) = crate::trace_walk::walk_core(
        be, &stream, &reg_addr_d, &ts_d, &_value_d, &init_value_d, init_ts, total, nbins,
    )?;
    let nr = num_rows as u64;
    let stride = total_ct as u64;
    let mut hist = stream.alloc_zeros::<u64>(total_ct * HIST_COPIES as usize)?;
    let n_u64 = total as u64;
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_memw_reg_routed)
            .arg(&n_u64)
            .arg(&ts_d)
            .arg(&old_ts_d)
            .arg(&row_index_d)
            .arg(&nr)
            .arg(&HIST_COPIES)
            .arg(&stride)
            .arg(&mut hist)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    unsafe {
        stream
            .launch_builder(&be.bitwise_hist_reduce)
            .arg(&stride)
            .arg(&HIST_COPIES)
            .arg(&hist)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(total_ct as u32))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}
