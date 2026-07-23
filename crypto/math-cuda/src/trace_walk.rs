//! On-GPU register memory-model walk (see `kernels/trace_walk.cu`).
//!
//! Recovers each register access's predecessor `(old_value, old_ts)` at its
//! register address in parallel — the device analog of the sequential CPU walk
//! (`prover/.../trace_builder.rs::walk_register_accesses`, the bit-for-bit
//! reference). Correctness-first stable counting-sort group-by + predecessor link.
//! `walk_core` is the device-resident core (the future MEMW_R fill reads its output
//! in place); `gpu_walk_registers` is the host-in/host-out entry used by the parity
//! test.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{Backend, backend};

/// Input elements per segment (one block per segment). Balances the offsets kernel's
/// per-thread segment loop against the scatter's per-block serial pass.
const WALK_SEG_SIZE: u64 = 4096;
/// Histogram block size (threads cooperate over a segment via grid-stride).
const WALK_HIST_BLOCK: u32 = 256;

/// Device-resident register walk: given device SoA `(key, ts, value)` and the
/// per-bucket seed `(init_value, init_ts)`, recover `(old_value, old_ts)` on device.
/// Returns the two device buffers (length `n`, indexed by original access), left
/// resident (unsynchronized on `stream`) so a downstream fill can consume them with
/// no host round-trip. `nbins` bounds the bucket key (register word address < nbins).
///
/// `pub(crate)` so `trace_cpu`'s combined walk+fill (`gpu_walk_and_fill_memw_register`)
/// can feed the resident output straight into the MEMW_R fill on the same stream.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walk_core(
    be: &Backend,
    stream: &Arc<CudaStream>,
    key_d: &CudaSlice<u32>,
    ts_d: &CudaSlice<u64>,
    value_d: &CudaSlice<u64>,
    init_value_d: &CudaSlice<u64>,
    init_ts: u64,
    n: usize,
    nbins: u32,
) -> Result<(CudaSlice<u64>, CudaSlice<u64>)> {
    let nb = nbins as usize;
    let seg = (n as u64).div_ceil(WALK_SEG_SIZE); // segments == grid.x for hist/scatter
    let n_u64 = n as u64;
    let seg_size = WALK_SEG_SIZE;

    let mut seg_hist = stream.alloc_zeros::<u64>(seg as usize * nb)?;
    let mut global_off = stream.alloc_zeros::<u64>(nb)?;
    let mut perm = stream.alloc_zeros::<u64>(n)?;
    let mut old_value = stream.alloc_zeros::<u64>(n)?;
    let mut old_ts = stream.alloc_zeros::<u64>(n)?;

    // 1. Per-segment histogram (dynamic shared u32[nbins]).
    unsafe {
        stream
            .launch_builder(&be.walk_seg_hist)
            .arg(key_d)
            .arg(&n_u64)
            .arg(&nbins)
            .arg(&seg_size)
            .arg(&mut seg_hist)
            .launch(LaunchConfig {
                grid_dim: (seg as u32, 1, 1),
                block_dim: (WALK_HIST_BLOCK, 1, 1),
                shared_mem_bytes: (nb * std::mem::size_of::<u32>()) as u32,
            })?;
    }

    // 2. Bucket base offsets + per-segment prefix (single block; thread per bucket).
    unsafe {
        stream
            .launch_builder(&be.walk_seg_offsets)
            .arg(&mut seg_hist)
            .arg(&seg)
            .arg(&nbins)
            .arg(&mut global_off)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (nbins, 1, 1),
                shared_mem_bytes: (nb * std::mem::size_of::<u64>()) as u32,
            })?;
    }

    // 3. Stable scatter into `perm` (block per segment; only lane 0 scatters, in order).
    unsafe {
        stream
            .launch_builder(&be.walk_seg_scatter)
            .arg(key_d)
            .arg(&n_u64)
            .arg(&nbins)
            .arg(&seg_size)
            .arg(&mut seg_hist)
            .arg(&mut perm)
            .launch(LaunchConfig {
                grid_dim: (seg as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    // 4. Predecessor link (one thread per grouped position).
    unsafe {
        stream
            .launch_builder(&be.walk_link)
            .arg(&perm)
            .arg(key_d)
            .arg(ts_d)
            .arg(value_d)
            .arg(&global_off)
            .arg(init_value_d)
            .arg(&init_ts)
            .arg(&n_u64)
            .arg(&mut old_value)
            .arg(&mut old_ts)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    Ok((old_value, old_ts))
}

/// Host-in / host-out register walk: upload the access SoA, walk on device, return
/// `(old_value, old_ts)` in original access order. The parity oracle for
/// `walk_register_accesses`. `key` is the register word address per access (< nbins);
/// `init_value[b]` is bucket `b`'s seed value (all seed timestamps `init_ts`).
pub fn gpu_walk_registers(
    key: &[u32],
    ts: &[u64],
    value: &[u64],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let n = key.len();
    assert_eq!(ts.len(), n);
    assert_eq!(value.len(), n);
    assert_eq!(init_value.len(), nbins as usize);
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let be = backend()?;
    let stream = be.next_stream();
    let key_d = stream.clone_htod(key)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let init_value_d = stream.clone_htod(init_value)?;

    let (old_value, old_ts) = walk_core(
        be,
        &stream,
        &key_d,
        &ts_d,
        &value_d,
        &init_value_d,
        init_ts,
        n,
        nbins,
    )?;

    let ov = stream.clone_dtoh(&old_value)?;
    let ot = stream.clone_dtoh(&old_ts)?;
    stream.synchronize()?;
    Ok((ov, ot))
}

/// IS_HALFWORD histogram width: one bin per possible ts-delta `ts_lo-old_ts_lo-1`.
const IS_HALF_BINS: usize = 1 << 16;
/// Scan tuning: aim for ~this many flags per block, capping the block count so the
/// spine (single block) scans all block totals in one pass.
const SCAN_TARGET_EPB: u64 = 8192;
const SCAN_MAX_BLOCKS: u64 = 1024;

/// P1: emit the REGISTER-access stream on device from the resident cpu_op fields (packed decode +
/// rv1/rv2/rvd/next_pc), the device analog of `trace_builder::emit_register_accesses` over all ops.
/// Returns `(reg_addr, ts, value, is_read, row_index)` in op-order (within-op M1/M3/M5/PC); emitting
/// accesses carry their compacted `row_index`, the implicit PC write carries `-1`. Feeds the device
/// register walk with NO host collection/upload of the accesses. `ts = i*4+4`.
#[allow(clippy::type_complexity)]
pub fn gpu_emit_register_accesses(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
) -> Result<(Vec<u32>, Vec<u64>, Vec<u64>, Vec<u8>, Vec<i64>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;
    let (reg_addr, ts_out, value, is_read, row_index, total) =
        emit_register_accesses_dev(be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n)?;
    let reg_addr_h = stream.clone_dtoh(&reg_addr)?;
    let ts_h = stream.clone_dtoh(&ts_out)?;
    let value_h = stream.clone_dtoh(&value)?;
    let is_read_h = stream.clone_dtoh(&is_read)?;
    let row_index_h = stream.clone_dtoh(&row_index)?;
    stream.synchronize()?;
    Ok((
        reg_addr_h[..total].to_vec(),
        ts_h[..total].to_vec(),
        value_h[..total].to_vec(),
        is_read_h[..total].to_vec(),
        row_index_h[..total].to_vec(),
    ))
}

/// Device core of the register-access emitter: from the resident cpu_op device buffers, produce the
/// access streams AS DEVICE BUFFERS (reg_addr, ts, value, is_read, row_index) + total count — no
/// download. The resident register walk (P2) feeds these to `walk_core` + the MEMW_R fill in place.
#[allow(clippy::type_complexity)]
pub(crate) fn emit_register_accesses_dev(
    be: &Backend,
    stream: &Arc<CudaStream>,
    pk_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    rv2_d: &CudaSlice<u64>,
    rvd_d: &CudaSlice<u64>,
    npc_d: &CudaSlice<u64>,
    n: usize,
) -> Result<(
    CudaSlice<u32>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u8>,
    CudaSlice<i64>,
    usize,
)> {
    let n_u64 = n as u64;
    let mut acc_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut emit_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.reg_access_counts)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(&mut acc_cnt)
                .arg(&mut emit_cnt)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (acc_off, total_acc) = excl_scan(be, stream, &acc_cnt, n.max(1))?;
    let (row_base, _total_emit) = excl_scan(be, stream, &emit_cnt, n.max(1))?;
    let total = total_acc as usize;

    let mut reg_addr = stream.alloc_zeros::<u32>(total.max(1))?;
    let mut ts_out = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut value = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut is_read = stream.alloc_zeros::<u8>(total.max(1))?;
    let mut row_index = stream.alloc_zeros::<i64>(total.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.reg_access_scatter)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(rv1_d)
                .arg(rv2_d)
                .arg(rvd_d)
                .arg(npc_d)
                .arg(&acc_off)
                .arg(&row_base)
                .arg(&mut reg_addr)
                .arg(&mut ts_out)
                .arg(&mut value)
                .arg(&mut is_read)
                .arg(&mut row_index)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    Ok((reg_addr, ts_out, value, is_read, row_index, total))
}

/// P1-ecall: register-access emission that INTERLEAVES the (host-captured) ecall accesses at their
/// op's timeline position. Same as [`emit_register_accesses_dev`], but each op reserves extra slots
/// for its ecall accesses (grouped by op via `ecall_op_index`, non-decreasing) and the scatter writes
/// them right after the op's regular accesses + PC write. The ecall accesses are EMITTING MEMW_R
/// candidates (`row_index >= 0`) — the routed histogram kernel applies the register ts-delta filter,
/// matching the CPU's `is_register_op` routing of ecall MemwOperations into `register_rows`; only the
/// implicit PC write stays non-emitting (`row_index = -1`). The stable-by-bin walk chains predecessors
/// in timeline order, so each ecall access's `old_ts` is its true predecessor. Uploads only the tiny
/// ecall arrays.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_register_accesses_with_ecall_dev(
    be: &Backend,
    stream: &Arc<CudaStream>,
    pk_d: &CudaSlice<u64>,
    rv1_d: &CudaSlice<u64>,
    rv2_d: &CudaSlice<u64>,
    rvd_d: &CudaSlice<u64>,
    npc_d: &CudaSlice<u64>,
    n: usize,
    ecall_op_index: &[u32],
    ecall_reg_addr: &[u32],
    ecall_ts: &[u64],
    ecall_value: &[u64],
    ecall_is_read: &[u8],
) -> Result<(
    CudaSlice<u32>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u8>,
    CudaSlice<i64>,
    usize,
)> {
    let n_u64 = n as u64;
    let m = ecall_op_index.len();
    let eidx_d = stream.clone_htod(ecall_op_index)?;
    let ereg_d = stream.clone_htod(ecall_reg_addr)?;
    let ets_d = stream.clone_htod(ecall_ts)?;
    let eval_d = stream.clone_htod(ecall_value)?;
    let eisr_d = stream.clone_htod(ecall_is_read)?;

    // Per-op ecall counts (scatter-add), then offsets.
    let mut ecall_op_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if m > 0 {
        unsafe {
            stream
                .launch_builder(&be.reg_ecall_op_counts)
                .arg(&(m as u64))
                .arg(&eidx_d)
                .arg(&mut ecall_op_cnt)
                .launch(LaunchConfig::for_num_elems(m as u32))?;
        }
    }
    let (ecall_op_off, _tot_ecall) = excl_scan(be, stream, &ecall_op_cnt, n.max(1))?;

    let mut acc_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut emit_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.reg_access_counts_ecall)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(&ecall_op_cnt)
                .arg(&mut acc_cnt)
                .arg(&mut emit_cnt)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (acc_off, total_acc) = excl_scan(be, stream, &acc_cnt, n.max(1))?;
    let (row_base, _total_emit) = excl_scan(be, stream, &emit_cnt, n.max(1))?;
    let total = total_acc as usize;

    let mut reg_addr = stream.alloc_zeros::<u32>(total.max(1))?;
    let mut ts_out = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut value = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut is_read = stream.alloc_zeros::<u8>(total.max(1))?;
    let mut row_index = stream.alloc_zeros::<i64>(total.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.reg_access_scatter_ecall)
                .arg(&n_u64)
                .arg(pk_d)
                .arg(rv1_d)
                .arg(rv2_d)
                .arg(rvd_d)
                .arg(npc_d)
                .arg(&acc_off)
                .arg(&row_base)
                .arg(&ecall_op_cnt)
                .arg(&ecall_op_off)
                .arg(&ereg_d)
                .arg(&ets_d)
                .arg(&eval_d)
                .arg(&eisr_d)
                .arg(&mut reg_addr)
                .arg(&mut ts_out)
                .arg(&mut value)
                .arg(&mut is_read)
                .arg(&mut row_index)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    Ok((reg_addr, ts_out, value, is_read, row_index, total))
}

/// Device-emitted LOAD/STORE memory accesses (P1): per-byte access streams + per-op metadata,
/// built on device from the resident cpu_ops — the device analog of the memw byte-access host prep.
pub struct DeviceMemAccesses {
    /// Per-byte-access (Σ width over load/store ops): address, timestamp, byte value, op row, byte offset.
    pub addr: Vec<u64>,
    pub ts: Vec<u64>,
    pub val: Vec<u64>,
    pub op_row: Vec<u64>,
    pub byte_off: Vec<u32>,
    /// Per-op (one per load/store op, compacted): base address, timestamp, is_read, width, signed, value word.
    pub base: Vec<u64>,
    pub op_ts: Vec<u64>,
    pub is_read: Vec<u32>,
    pub width: Vec<u32>,
    pub signed: Vec<u32>,
    pub value_word: Vec<u64>,
}

/// P1: emit the LOAD/STORE memory accesses on device from the resident cpu_op fields
/// (packed decode + res + rvd + rv2). Per load/store op: `width` byte-accesses (addr=res+j, ts,
/// byte j of value) + per-op metadata. `ts = i*4+4`. Feeds the device memory walk with NO host
/// byte-access collection. (init_value seeding from the image is a separate concern.)
pub fn gpu_emit_memory_accesses(
    packed: &[u64],
    res: &[u64],
    rvd: &[u64],
    rv2: &[u64],
) -> Result<DeviceMemAccesses> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let res_d = stream.clone_htod(res)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let rv2_d = stream.clone_htod(rv2)?;

    let mut ls_flag = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut byte_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.memacc_counts)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&mut ls_flag)
                .arg(&mut byte_cnt)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (op_off, num_ops_u64) = excl_scan(be, &stream, &ls_flag, n.max(1))?;
    let (byte_base, total_bytes_u64) = excl_scan(be, &stream, &byte_cnt, n.max(1))?;
    let num_ops = num_ops_u64 as usize;
    let total = total_bytes_u64 as usize;

    let mut base = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut op_ts = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut is_read = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut width_out = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut signed_out = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut value_word = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut addr = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut ts_a = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut val_a = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut op_row = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut byte_off = stream.alloc_zeros::<u32>(total.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.memacc_emit)
                .arg(&n_u64)
                .arg(&pk_d)
                .arg(&res_d)
                .arg(&rvd_d)
                .arg(&rv2_d)
                .arg(&op_off)
                .arg(&byte_base)
                .arg(&mut base)
                .arg(&mut op_ts)
                .arg(&mut is_read)
                .arg(&mut width_out)
                .arg(&mut signed_out)
                .arg(&mut value_word)
                .arg(&mut addr)
                .arg(&mut ts_a)
                .arg(&mut val_a)
                .arg(&mut op_row)
                .arg(&mut byte_off)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let out = DeviceMemAccesses {
        addr: stream.clone_dtoh(&addr)?[..total].to_vec(),
        ts: stream.clone_dtoh(&ts_a)?[..total].to_vec(),
        val: stream.clone_dtoh(&val_a)?[..total].to_vec(),
        op_row: stream.clone_dtoh(&op_row)?[..total].to_vec(),
        byte_off: stream.clone_dtoh(&byte_off)?[..total].to_vec(),
        base: stream.clone_dtoh(&base)?[..num_ops].to_vec(),
        op_ts: stream.clone_dtoh(&op_ts)?[..num_ops].to_vec(),
        is_read: stream.clone_dtoh(&is_read)?[..num_ops].to_vec(),
        width: stream.clone_dtoh(&width_out)?[..num_ops].to_vec(),
        signed: stream.clone_dtoh(&signed_out)?[..num_ops].to_vec(),
        value_word: stream.clone_dtoh(&value_word)?[..num_ops].to_vec(),
    };
    stream.synchronize()?;
    Ok(out)
}

/// Image-on-device core: for each device `addr_d[i]`, look up the initial-image byte via binary
/// search over the sorted image (`img_addr_d` ascending, `img_val_d`), returning a resident
/// `init_value` buffer (0 where absent). Shared by the resident memory walk (init_value) and PAGE.
pub(crate) fn image_lookup_dev(
    be: &Backend,
    stream: &Arc<CudaStream>,
    addr_d: &CudaSlice<u64>,
    img_addr_d: &CudaSlice<u64>,
    img_val_d: &CudaSlice<u64>,
    n: usize,
    n_img: usize,
) -> Result<CudaSlice<u64>> {
    let mut out = stream.alloc_zeros::<u64>(n.max(1))?;
    if n > 0 {
        let (n_u64, ni_u64) = (n as u64, n_img as u64);
        unsafe {
            stream
                .launch_builder(&be.image_lookup)
                .arg(&n_u64)
                .arg(addr_d)
                .arg(img_addr_d)
                .arg(img_val_d)
                .arg(&ni_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    Ok(out)
}

/// Host-in/out image lookup for parity tests: `img_addr` must be ascending. Returns the image byte
/// per `addr` (0 if absent) — the device analog of `image.get(addr).unwrap_or(0)`.
pub fn gpu_image_lookup(addr: &[u64], img_addr: &[u64], img_val: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(img_addr.len(), img_val.len());
    let be = backend()?;
    let stream = be.next_stream();
    let addr_d = stream.clone_htod(addr)?;
    let ia_d = stream.clone_htod(img_addr)?;
    let iv_d = stream.clone_htod(img_val)?;
    let out = image_lookup_dev(be, &stream, &addr_d, &ia_d, &iv_d, addr.len(), img_addr.len())?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host[..addr.len()].to_vec())
}

/// Device exclusive prefix scan of a 0/1 `flag` array of length `n`. Returns the
/// per-element exclusive prefix (device) and the grand total (host). Two-level
/// (per-block totals → spine scan → per-block write); see `trace_walk.cu`.
pub(crate) fn excl_scan(
    be: &Backend,
    stream: &Arc<CudaStream>,
    flag_d: &CudaSlice<u32>,
    n: usize,
) -> Result<(CudaSlice<u64>, u64)> {
    let n_u64 = n as u64;
    let nblocks = n_u64.div_ceil(SCAN_TARGET_EPB).clamp(1, SCAN_MAX_BLOCKS);
    let epb = n_u64.div_ceil(nblocks);
    let mut block_tot = stream.alloc_zeros::<u64>(nblocks as usize)?;
    let mut total_out = stream.alloc_zeros::<u64>(1)?;
    let mut excl = stream.alloc_zeros::<u64>(n)?;
    let seg_cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let one_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.scan_reduce)
            .arg(flag_d)
            .arg(&n_u64)
            .arg(&epb)
            .arg(&mut block_tot)
            .launch(seg_cfg)?;
    }
    unsafe {
        stream
            .launch_builder(&be.scan_spine)
            .arg(&mut block_tot)
            .arg(&nblocks)
            .arg(&mut total_out)
            .launch(one_cfg)?;
    }
    unsafe {
        stream
            .launch_builder(&be.scan_write_excl)
            .arg(flag_d)
            .arg(&n_u64)
            .arg(&epb)
            .arg(&block_tot)
            .arg(&mut excl)
            .launch(seg_cfg)?;
    }
    let total = stream.clone_dtoh(&total_out)?;
    Ok((excl, total[0]))
}

/// Shared walk → route → compact core (the "option A" path). The resident SoA and
/// the GLOBAL compacted `row_index_d` (`0..num_rows`, or −1 for non-rows) stay in
/// VRAM so the caller can fill one or many capped MEMW_R chunk tables from a single
/// walk; `is_half_counts`/`fallback` are the host-side artifacts. `reg_addr` is the
/// register word address per access (walk bucket key < nbins and the fill's address
/// column); `emits_row[i]=0` marks a timeline-only access (implicit PC write).
struct RouteCore {
    reg_addr_d: CudaSlice<u32>,
    ts_d: CudaSlice<u64>,
    value_d: CudaSlice<u64>,
    is_read_d: CudaSlice<u8>,
    old_value_d: CudaSlice<u64>,
    old_ts_d: CudaSlice<u64>,
    row_index_d: CudaSlice<i64>,
    n: usize,
    num_rows: usize,
    is_half_counts: Vec<u64>,
    fallback: Vec<[u64; 6]>,
}

/// Walk → route → compact on `stream` (assumes `n > 0`). `init_value[b]` seeds bucket
/// `b` at `init_ts`. Leaves everything resident (unsynchronized) for a downstream fill.
#[allow(clippy::too_many_arguments)]
fn route_core_on(
    be: &Backend,
    stream: &Arc<CudaStream>,
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    emits_row: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
) -> Result<RouteCore> {
    let n = reg_addr.len();
    assert!(n > 0);
    assert_eq!(ts.len(), n);
    assert_eq!(value.len(), n);
    assert_eq!(is_read.len(), n);
    assert_eq!(emits_row.len(), n);

    // Upload the access SoA once; shared by walk, route, fill, and gather.
    let reg_addr_d = stream.clone_htod(reg_addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let is_read_d = stream.clone_htod(is_read)?;
    let emits_row_d = stream.clone_htod(emits_row)?;
    route_core_from_device(
        be, stream, reg_addr_d, ts_d, value_d, is_read_d, emits_row_d, init_value, init_ts, nbins, n,
    )
}

/// Route → compact → fill-prep over an ALREADY-RESIDENT access stream (walk + route + fallback
/// gather). Shared by the host-uploaded regular path ([`route_core_on`]) and the device-emitted
/// ecall path ([`gpu_walk_route_memw_register_ecall_chunked`]) so both produce identical MEMW_R
/// routing. Consumes the access-SoA device buffers (moved into the returned `RouteCore`).
/// `emits_row_d[i]=0` marks a timeline-only access (PC write); ecall + regular accesses are 1.
#[allow(clippy::too_many_arguments)]
fn route_core_from_device(
    be: &Backend,
    stream: &Arc<CudaStream>,
    reg_addr_d: CudaSlice<u32>,
    ts_d: CudaSlice<u64>,
    value_d: CudaSlice<u64>,
    is_read_d: CudaSlice<u8>,
    emits_row_d: CudaSlice<u8>,
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    n: usize,
) -> Result<RouteCore> {
    assert!(n > 0);
    assert_eq!(init_value.len(), nbins as usize);
    let init_value_d = stream.clone_htod(init_value)?;

    // 1. Walk → resident (old_value, old_ts).
    let (old_value_d, old_ts_d) = walk_core(
        be,
        stream,
        &reg_addr_d,
        &ts_d,
        &value_d,
        &init_value_d,
        init_ts,
        n,
        nbins,
    )?;

    // 2. Route each access → MEMW_R vs fallback flags.
    let n_u64 = n as u64;
    let mut flag_memw = stream.alloc_zeros::<u32>(n)?;
    let mut flag_fb = stream.alloc_zeros::<u32>(n)?;
    unsafe {
        stream
            .launch_builder(&be.memw_route_flags)
            .arg(&n_u64)
            .arg(&ts_d)
            .arg(&old_ts_d)
            .arg(&emits_row_d)
            .arg(&mut flag_memw)
            .arg(&mut flag_fb)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    // 3. Compact each partition into contiguous positions.
    let (excl_memw, total_memw) = excl_scan(be, stream, &flag_memw, n)?;
    let (excl_fb, total_fb) = excl_scan(be, stream, &flag_fb, n)?;
    let num_rows = total_memw as usize;

    // 4. Global compacted MEMW_R row index (−1 elsewhere).
    let mut row_index_d = stream.alloc_zeros::<i64>(n)?;
    unsafe {
        stream
            .launch_builder(&be.memw_rowindex_from_excl)
            .arg(&n_u64)
            .arg(&flag_memw)
            .arg(&excl_memw)
            .arg(&mut row_index_d)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    // 5. IS_HALFWORD delta histogram (one +1 per MEMW_R row).
    let mut hist = stream.alloc_zeros::<u64>(IS_HALF_BINS)?;
    unsafe {
        stream
            .launch_builder(&be.memw_is_half_hist)
            .arg(&n_u64)
            .arg(&flag_memw)
            .arg(&ts_d)
            .arg(&old_ts_d)
            .arg(&mut hist)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let is_half_counts = stream.clone_dtoh(&hist)?;

    // 6. Gather the rare fallback subset (emit order) for the host.
    let fallback = if total_fb == 0 {
        Vec::new()
    } else {
        let mut fb_out = stream.alloc_zeros::<u64>(total_fb as usize * 6)?;
        unsafe {
            stream
                .launch_builder(&be.memw_fb_gather)
                .arg(&n_u64)
                .arg(&flag_fb)
                .arg(&excl_fb)
                .arg(&reg_addr_d)
                .arg(&ts_d)
                .arg(&value_d)
                .arg(&old_value_d)
                .arg(&old_ts_d)
                .arg(&is_read_d)
                .arg(&mut fb_out)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        let flat = stream.clone_dtoh(&fb_out)?;
        flat.chunks_exact(6)
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
            .collect()
    };

    Ok(RouteCore {
        reg_addr_d,
        ts_d,
        value_d,
        is_read_d,
        old_value_d,
        old_ts_d,
        row_index_d,
        n,
        num_rows,
        is_half_counts,
        fallback,
    })
}

/// Fill one capped MEMW_R chunk table for global rows `[row_lo, row_hi)` from `core`:
/// localize the global row index to the chunk, then the shared `memw_register_fill`.
/// Returns the resident `[height * NCOLS]` buffer, `height =
/// (row_hi-row_lo).next_power_of_two().max(4)` (matching the CPU chunk table height).
fn fill_chunk_on(
    be: &Backend,
    stream: &Arc<CudaStream>,
    core: &RouteCore,
    row_lo: usize,
    row_hi: usize,
) -> Result<CudaSlice<u64>> {
    let ncols = crate::trace_cpu::MEMW_REGISTER_NCOLS;
    let height = (row_hi - row_lo).next_power_of_two().max(4);
    let mut buf = stream.alloc_zeros::<u64>(height * ncols)?;
    if core.n == 0 {
        return Ok(buf);
    }
    let n_u64 = core.n as u64;
    let lo_i = row_lo as i64;
    let hi_i = row_hi as i64;
    let mut local = stream.alloc_zeros::<i64>(core.n)?;
    unsafe {
        stream
            .launch_builder(&be.memw_rowindex_localize)
            .arg(&n_u64)
            .arg(&core.row_index_d)
            .arg(&lo_i)
            .arg(&hi_i)
            .arg(&mut local)
            .launch(LaunchConfig::for_num_elems(core.n as u32))?;
    }
    let ncols_u32 = ncols as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&core.reg_addr_d)
            .arg(&core.ts_d)
            .arg(&core.value_d)
            .arg(&core.is_read_d)
            .arg(&local)
            .arg(&core.old_value_d)
            .arg(&core.old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(core.n as u32))?;
    }
    Ok(buf)
}

/// Device MEMW_R build (production): one on-GPU walk feeds every capped MEMW_R chunk
/// table (`max_rows_per_table` = the per-table row cap). Returns the resident chunk
/// buffers (each `[height*NCOLS]`, left in VRAM for the LDE), the real MEMW_R row
/// count, the IS_HALFWORD delta counts (`[65536]`, merge into the BITWISE histogram),
/// and the rare fallback subset (`[reg_addr, ts, value, old_value, old_ts, is_read]`,
/// emit order) the host routes to aligned/general. Empty input → one zeroed table.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gpu_walk_route_memw_register_chunked(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    emits_row: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    max_rows_per_table: usize,
) -> Result<(Vec<CudaSlice<u64>>, usize, Vec<u64>, Vec<[u64; 6]>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::MEMW_REGISTER_NCOLS;
    if reg_addr.is_empty() {
        let buf = stream.alloc_zeros::<u64>(4 * ncols)?;
        stream.synchronize()?;
        return Ok((vec![buf], 0, vec![0u64; IS_HALF_BINS], Vec::new()));
    }
    let core = route_core_on(
        be, &stream, reg_addr, ts, value, is_read, emits_row, init_value, init_ts, nbins,
    )?;
    let num_rows = core.num_rows;
    let n_chunks = num_rows.div_ceil(max_rows_per_table).max(1);
    let mut bufs = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let lo = c * max_rows_per_table;
        let hi = ((c + 1) * max_rows_per_table).min(num_rows);
        bufs.push(fill_chunk_on(be, &stream, &core, lo, hi)?);
    }
    stream.synchronize()?;
    Ok((bufs, num_rows, core.is_half_counts, core.fallback))
}

/// Device MEMW_R build WITH ecall interleaving (production, precompile runs). Emits the register
/// access stream on device from the resident cpu_op fields AND the (host-captured) COMMIT/KECCAK/ECSM
/// register accesses interleaved at their op's timeline position, then walks/routes/fills exactly like
/// [`gpu_walk_route_memw_register_chunked`]. The route step (`route_core_from_device`) classifies every
/// emitting access — regular and ecall — into MEMW_R vs aligned/general fallback via the same ts-delta
/// condition the CPU's `is_register_op` uses, so the resident tables + fallbacks are bit-identical to
/// the sequential `collect_ops_from_cpu` register rows. Returns the resident chunk buffers, MEMW_R row
/// count, IS_HALFWORD delta counts, and the fallback subset (emit order). Empty input → one zeroed table.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gpu_walk_route_memw_register_ecall_chunked(
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
    max_rows_per_table: usize,
) -> Result<(Vec<CudaSlice<u64>>, usize, Vec<u64>, Vec<[u64; 6]>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::MEMW_REGISTER_NCOLS;
    let n = packed.len();
    let empty = || -> Result<_> {
        let buf = stream.alloc_zeros::<u64>(4 * ncols)?;
        stream.synchronize()?;
        Ok((vec![buf], 0usize, vec![0u64; IS_HALF_BINS], Vec::new()))
    };
    if n == 0 {
        return empty();
    }
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;
    let (reg_addr_d, ts_d, value_d, is_read_d, row_index_d, total) =
        emit_register_accesses_with_ecall_dev(
            be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n, ecall_op_index, ecall_reg_addr,
            ecall_ts, ecall_value, ecall_is_read,
        )?;
    if total == 0 {
        return empty();
    }
    // Derive the route mask from the emitter's compacted row index (>=0 => MEMW_R candidate).
    let mut emits_row_d = stream.alloc_zeros::<u8>(total)?;
    let total_u64 = total as u64;
    unsafe {
        stream
            .launch_builder(&be.rowindex_to_emits)
            .arg(&total_u64)
            .arg(&row_index_d)
            .arg(&mut emits_row_d)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let core = route_core_from_device(
        be, &stream, reg_addr_d, ts_d, value_d, is_read_d, emits_row_d, init_value, init_ts, nbins,
        total,
    )?;
    let num_rows = core.num_rows;
    let n_chunks = num_rows.div_ceil(max_rows_per_table).max(1);
    let mut bufs = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let lo = c * max_rows_per_table;
        let hi = ((c + 1) * max_rows_per_table).min(num_rows);
        bufs.push(fill_chunk_on(be, &stream, &core, lo, hi)?);
    }
    stream.synchronize()?;
    Ok((bufs, num_rows, core.is_half_counts, core.fallback))
}

/// C2-c1: the device REGISTER FINAL STATE snapshot — per register word-address (`0..init_value.len()`),
/// the `(value, ts)` of the last access (max ts), the device analog of `RegisterState::to_final_state_map`.
/// Emits the register-access stream on device (regular M1/M3/M5 + interleaved ecall accesses), then
/// reduces per address. `init_value[a]` seeds address `a` at `init_ts` (never-accessed registers keep
/// init). Returns `(out_value[naddr], out_ts[naddr])`. `naddr = init_value.len()` (= REG_WALK_NBINS).
#[allow(clippy::too_many_arguments)]
pub fn gpu_register_final_snapshot(
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
    // C2-c2: x254 (commit index, addr 508) is tracked outside the walk. `commit_flag[i]=1` on
    // ecall-commit ops, `commit_count[i]` their count (0 elsewhere); `start_commit_index` = the x254
    // init. final x254 = start + Σ count, ts = last commit op ts (or init_ts if there are no commits).
    commit_flag: &[u8],
    commit_count: &[u64],
    start_commit_index: u64,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let naddr = init_value.len();
    let mut out_val = stream.alloc_zeros::<u64>(naddr)?;
    let mut out_ts = stream.alloc_zeros::<u64>(naddr)?;
    let mut max_ts = stream.alloc_zeros::<u64>(naddr)?;
    let init_value_d = stream.clone_htod(init_value)?;
    let naddr_u64 = naddr as u64;
    unsafe {
        stream
            .launch_builder(&be.reg_final_seed)
            .arg(&naddr_u64)
            .arg(&init_value_d)
            .arg(&init_ts)
            .arg(&mut out_val)
            .arg(&mut out_ts)
            .arg(&mut max_ts)
            .launch(LaunchConfig::for_num_elems(naddr as u32))?;
    }
    let n = packed.len();
    if n > 0 {
        let pk_d = stream.clone_htod(packed)?;
        let rv1_d = stream.clone_htod(rv1)?;
        let rv2_d = stream.clone_htod(rv2)?;
        let rvd_d = stream.clone_htod(rvd)?;
        let npc_d = stream.clone_htod(next_pc)?;
        let (reg_addr_d, ts_d, value_d, _is_read_d, _row_index_d, total) =
            emit_register_accesses_with_ecall_dev(
                be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n, ecall_op_index, ecall_reg_addr,
                ecall_ts, ecall_value, ecall_is_read,
            )?;
        if total > 0 {
            let total_u64 = total as u64;
            let cfg = LaunchConfig::for_num_elems(total as u32);
            unsafe {
                stream
                    .launch_builder(&be.reg_final_maxts)
                    .arg(&total_u64)
                    .arg(&reg_addr_d)
                    .arg(&ts_d)
                    .arg(&mut max_ts)
                    .launch(cfg)?;
            }
            unsafe {
                stream
                    .launch_builder(&be.reg_final_gather)
                    .arg(&total_u64)
                    .arg(&reg_addr_d)
                    .arg(&ts_d)
                    .arg(&value_d)
                    .arg(&max_ts)
                    .arg(&mut out_val)
                    .arg(&mut out_ts)
                    .launch(cfg)?;
            }
        }
    }
    // C2-c2: x254 commit index (separate from the walk). Sum commit counts + track the last commit ts.
    let (mut x254_total, mut x254_last_ts) = (0u64, 0u64);
    if n > 0 && !commit_flag.is_empty() {
        let cf_d = stream.clone_htod(commit_flag)?;
        let cc_d = stream.clone_htod(commit_count)?;
        let mut total_d = stream.alloc_zeros::<u64>(1)?;
        let mut lastts_d = stream.alloc_zeros::<u64>(1)?;
        let n_u64 = n as u64;
        unsafe {
            stream
                .launch_builder(&be.reg_x254_scan)
                .arg(&n_u64)
                .arg(&cf_d)
                .arg(&cc_d)
                .arg(&mut total_d)
                .arg(&mut lastts_d)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        x254_total = stream.clone_dtoh(&total_d)?[0];
        x254_last_ts = stream.clone_dtoh(&lastts_d)?[0];
    }
    let mut hv = stream.clone_dtoh(&out_val)?;
    let mut ht = stream.clone_dtoh(&out_ts)?;
    stream.synchronize()?;
    if naddr > 508 {
        hv[508] = start_commit_index.wrapping_add(x254_total);
        ht[508] = if x254_last_ts > 0 { x254_last_ts } else { init_ts };
    }
    Ok((hv, ht))
}

/// DIAGNOSTIC: return the device MEMW_R build's ROWS (not the packed table) — every emitting register
/// access that routed to MEMW_R, as `[reg_addr, ts, value, old_value, old_ts, is_read]` — plus the
/// fallback subset (same shape). Lets a host test compare the full row content (value/old_value, not
/// just the IS_HALF ts-delta projection) against the sequential `memw_register_rows` as multisets.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gpu_walk_route_memw_register_ecall_rows_host(
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
) -> Result<(Vec<[u64; 6]>, Vec<[u64; 6]>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let pk_d = stream.clone_htod(packed)?;
    let rv1_d = stream.clone_htod(rv1)?;
    let rv2_d = stream.clone_htod(rv2)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let npc_d = stream.clone_htod(next_pc)?;
    let (reg_addr_d, ts_d, value_d, is_read_d, row_index_d, total) =
        emit_register_accesses_with_ecall_dev(
            be, &stream, &pk_d, &rv1_d, &rv2_d, &rvd_d, &npc_d, n, ecall_op_index, ecall_reg_addr,
            ecall_ts, ecall_value, ecall_is_read,
        )?;
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut emits_row_d = stream.alloc_zeros::<u8>(total)?;
    let total_u64 = total as u64;
    unsafe {
        stream
            .launch_builder(&be.rowindex_to_emits)
            .arg(&total_u64)
            .arg(&row_index_d)
            .arg(&mut emits_row_d)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let core = route_core_from_device(
        be, &stream, reg_addr_d, ts_d, value_d, is_read_d, emits_row_d, init_value, init_ts, nbins,
        total,
    )?;
    // Download the per-access arrays + the route compaction index; keep the accesses that routed to
    // MEMW_R (row_index >= 0). Fallbacks come back already gathered in `core.fallback`.
    let reg_addr_h = stream.clone_dtoh(&core.reg_addr_d)?;
    let ts_h = stream.clone_dtoh(&core.ts_d)?;
    let value_h = stream.clone_dtoh(&core.value_d)?;
    let old_value_h = stream.clone_dtoh(&core.old_value_d)?;
    let old_ts_h = stream.clone_dtoh(&core.old_ts_d)?;
    let is_read_h = stream.clone_dtoh(&core.is_read_d)?;
    let row_index_h = stream.clone_dtoh(&core.row_index_d)?;
    stream.synchronize()?;
    let mut memw_rows = Vec::with_capacity(core.num_rows);
    for i in 0..core.n {
        if row_index_h[i] >= 0 {
            memw_rows.push([
                reg_addr_h[i] as u64,
                ts_h[i],
                value_h[i],
                old_value_h[i],
                old_ts_h[i],
                is_read_h[i] as u64,
            ]);
        }
    }
    Ok((memw_rows, core.fallback))
}

/// Host-returning single-table variant for byte-parity tests: fills one `max_rows`
/// matrix (the global row index doubles as the local index) and downloads it.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gpu_walk_route_memw_register_host(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    emits_row: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    max_rows: usize,
) -> Result<(Vec<u64>, usize, Vec<u64>, Vec<[u64; 6]>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let ncols = crate::trace_cpu::MEMW_REGISTER_NCOLS;
    if reg_addr.is_empty() {
        return Ok((
            vec![0u64; max_rows * ncols],
            0,
            vec![0u64; IS_HALF_BINS],
            Vec::new(),
        ));
    }
    let core = route_core_on(
        be, &stream, reg_addr, ts, value, is_read, emits_row, init_value, init_ts, nbins,
    )?;
    assert!(
        core.num_rows <= max_rows,
        "MEMW_R rows ({}) exceed max_rows ({max_rows})",
        core.num_rows
    );
    // Single table: global row index is already the local index (chunk starts at 0).
    let mut buf = stream.alloc_zeros::<u64>(max_rows * ncols)?;
    let n_u64 = core.n as u64;
    let ncols_u32 = ncols as u32;
    unsafe {
        stream
            .launch_builder(&be.memw_register_fill)
            .arg(&n_u64)
            .arg(&core.reg_addr_d)
            .arg(&core.ts_d)
            .arg(&core.value_d)
            .arg(&core.is_read_d)
            .arg(&core.row_index_d)
            .arg(&core.old_value_d)
            .arg(&core.old_ts_d)
            .arg(&ncols_u32)
            .arg(&mut buf)
            .launch(LaunchConfig::for_num_elems(core.n as u32))?;
    }
    let host_buf = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok((host_buf, core.num_rows, core.is_half_counts, core.fallback))
}

// =============================================================================
// Memory memory-model walk (Phase 2): stable LSD radix sort by 64-bit byte address
// (8 passes, 256-bin digits, reusing walk_seg_offsets) + predecessor link.
// =============================================================================

/// Device memory memory-model walk. Given per byte-access `addr`, `ts`, written `value`,
/// and per-access `init_value` (the initial-image byte, used to seed the first access to
/// each address at `old_ts = 0`), returns `(old_value, old_ts)` per access in original
/// order — the predecessor `(value, ts)` for that byte address, or `(init_value, 0)` for
/// the first access to an address. Accesses must be supplied in ts (emission) order; the
/// sort is stable so a sort by address alone yields (address, ts) order. Host-in/host-out
/// (parity oracle; the resident form follows once downstream stages consume it).
pub fn gpu_mem_walk(
    addr: &[u64],
    ts: &[u64],
    value: &[u64],
    init_value: &[u64],
) -> Result<(Vec<u64>, Vec<u64>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = addr.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    debug_assert_eq!(ts.len(), n);
    debug_assert_eq!(value.len(), n);
    debug_assert_eq!(init_value.len(), n);

    let addr_d = stream.clone_htod(addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let init_d = stream.clone_htod(init_value)?;

    let n_u64 = n as u64;
    let seg = n_u64.div_ceil(WALK_SEG_SIZE);
    let seg_size = WALK_SEG_SIZE;
    const NBINS: u32 = 256;

    let mut perm_a = stream.alloc_zeros::<u64>(n)?;
    let mut perm_b = stream.alloc_zeros::<u64>(n)?;
    unsafe {
        stream
            .launch_builder(&be.radix_iota)
            .arg(&mut perm_a)
            .arg(&n_u64)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    let mut seg_hist = stream.alloc_zeros::<u64>(seg as usize * NBINS as usize)?;
    let mut global_off = stream.alloc_zeros::<u64>(NBINS as usize)?;

    let hist_cfg = LaunchConfig {
        grid_dim: (seg as u32, 1, 1),
        block_dim: (WALK_HIST_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let off_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (NBINS, 1, 1),
        shared_mem_bytes: NBINS * std::mem::size_of::<u64>() as u32,
    };
    let scat_cfg = LaunchConfig {
        grid_dim: (seg as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // 8 stable passes over the 8 bytes of the address (LSD → MSD). src=perm_a, dst=perm_b,
    // swapped each pass, so perm_a holds the sorted order after the loop.
    for pass in 0..8u32 {
        let shift = pass * 8;
        unsafe {
            stream
                .launch_builder(&be.radix_seg_hist)
                .arg(&perm_a)
                .arg(&addr_d)
                .arg(&n_u64)
                .arg(&shift)
                .arg(&seg_size)
                .arg(&mut seg_hist)
                .launch(hist_cfg)?;
        }
        unsafe {
            stream
                .launch_builder(&be.walk_seg_offsets)
                .arg(&mut seg_hist)
                .arg(&seg)
                .arg(&NBINS)
                .arg(&mut global_off)
                .launch(off_cfg)?;
        }
        unsafe {
            stream
                .launch_builder(&be.radix_seg_scatter)
                .arg(&perm_a)
                .arg(&addr_d)
                .arg(&n_u64)
                .arg(&shift)
                .arg(&seg_size)
                .arg(&mut seg_hist)
                .arg(&mut perm_b)
                .launch(scat_cfg)?;
        }
        std::mem::swap(&mut perm_a, &mut perm_b);
    }

    let mut old_value = stream.alloc_zeros::<u64>(n)?;
    let mut old_ts = stream.alloc_zeros::<u64>(n)?;
    unsafe {
        stream
            .launch_builder(&be.mem_link)
            .arg(&perm_a)
            .arg(&addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&init_d)
            .arg(&n_u64)
            .arg(&mut old_value)
            .arg(&mut old_ts)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    let ov = stream.clone_dtoh(&old_value)?;
    let ot = stream.clone_dtoh(&old_ts)?;
    stream.synchronize()?;
    Ok((ov, ot))
}

// =============================================================================
// Device dedup (resident enabler for the deduped chips). Sort a permutation by the full
// (k0,k1,k2) op key via multi-word LSD radix, then segment-reduce equal keys into unique
// rows with summed multiplicity.
// =============================================================================

/// Stable LSD radix sort of a fresh permutation by the given u64 key arrays (`keys[0]` is
/// the least-significant word, sorted first). Returns the sorted permutation (device).
/// Reuses the radix hist/offsets/scatter kernels — each key word is 8 byte-passes.
fn radix_sort_perm(
    be: &Backend,
    stream: &Arc<CudaStream>,
    keys: &[&CudaSlice<u64>],
    n: usize,
) -> Result<CudaSlice<u64>> {
    let n_u64 = n as u64;
    let seg = n_u64.div_ceil(WALK_SEG_SIZE);
    let seg_size = WALK_SEG_SIZE;
    const NBINS: u32 = 256;
    let mut perm_a = stream.alloc_zeros::<u64>(n)?;
    let mut perm_b = stream.alloc_zeros::<u64>(n)?;
    unsafe {
        stream
            .launch_builder(&be.radix_iota)
            .arg(&mut perm_a)
            .arg(&n_u64)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let mut seg_hist = stream.alloc_zeros::<u64>(seg as usize * NBINS as usize)?;
    let mut global_off = stream.alloc_zeros::<u64>(NBINS as usize)?;
    let hist_cfg = LaunchConfig {
        grid_dim: (seg as u32, 1, 1),
        block_dim: (WALK_HIST_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let off_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (NBINS, 1, 1),
        shared_mem_bytes: NBINS * std::mem::size_of::<u64>() as u32,
    };
    let scat_cfg = LaunchConfig {
        grid_dim: (seg as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    for key in keys {
        for pass in 0..8u32 {
            let shift = pass * 8;
            unsafe {
                stream
                    .launch_builder(&be.radix_seg_hist)
                    .arg(&perm_a)
                    .arg(*key)
                    .arg(&n_u64)
                    .arg(&shift)
                    .arg(&seg_size)
                    .arg(&mut seg_hist)
                    .launch(hist_cfg)?;
            }
            unsafe {
                stream
                    .launch_builder(&be.walk_seg_offsets)
                    .arg(&mut seg_hist)
                    .arg(&seg)
                    .arg(&NBINS)
                    .arg(&mut global_off)
                    .launch(off_cfg)?;
            }
            unsafe {
                stream
                    .launch_builder(&be.radix_seg_scatter)
                    .arg(&perm_a)
                    .arg(*key)
                    .arg(&n_u64)
                    .arg(&shift)
                    .arg(&seg_size)
                    .arg(&mut seg_hist)
                    .arg(&mut perm_b)
                    .launch(scat_cfg)?;
            }
            std::mem::swap(&mut perm_a, &mut perm_b);
        }
    }
    Ok(perm_a) // even passes per key → result is in perm_a
}

/// One deduped chip's unique rows: parallel arrays of the three key words + multiplicity.
pub struct DeviceDedup {
    pub k0: Vec<u64>,
    pub k1: Vec<u64>,
    pub k2: Vec<u64>,
    pub mult: Vec<u64>,
}

/// Device dedup over a 3-word op key: collapse identical `(k0,k1,k2)` triples into unique
/// rows with summed multiplicity (the host `HashMap<Op,mult>` done on GPU). Sort by the full
/// key → mark run starts → exclusive-scan → emit one row per run. Output order is sorted
/// (not the host's HashMap order), which the order-independent LogUp chip buses accept.
pub fn gpu_dedup3(k0: &[u64], k1: &[u64], k2: &[u64]) -> Result<DeviceDedup> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = k0.len();
    if n == 0 {
        return Ok(DeviceDedup {
            k0: vec![],
            k1: vec![],
            k2: vec![],
            mult: vec![],
        });
    }
    debug_assert_eq!(k1.len(), n);
    debug_assert_eq!(k2.len(), n);
    let k0_d = stream.clone_htod(k0)?;
    let k1_d = stream.clone_htod(k1)?;
    let k2_d = stream.clone_htod(k2)?;
    let (out_k0, out_k1, out_k2, out_mult, _m) =
        dedup3_core(be, &stream, &k0_d, &k1_d, &k2_d, n)?;
    let dd = DeviceDedup {
        k0: stream.clone_dtoh(&out_k0)?,
        k1: stream.clone_dtoh(&out_k1)?,
        k2: stream.clone_dtoh(&out_k2)?,
        mult: stream.clone_dtoh(&out_mult)?,
    };
    stream.synchronize()?;
    Ok(dd)
}

/// Device-buffer core of [`gpu_dedup3`] (no host transfer): sort by full key → mark runs →
/// scan → emit. Returns the unique `(k0,k1,k2,mult)` device buffers and the unique count.
/// Used by the resident chip chains that keep extract→dedup→pack→fill entirely on device.
#[allow(clippy::type_complexity)]
pub(crate) fn dedup3_core(
    be: &Backend,
    stream: &Arc<CudaStream>,
    k0_d: &CudaSlice<u64>,
    k1_d: &CudaSlice<u64>,
    k2_d: &CudaSlice<u64>,
    n: usize,
) -> Result<(
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    usize,
)> {
    // Sort by the full key (LSD: k0 least significant).
    let perm = radix_sort_perm(be, stream, &[k0_d, k1_d, k2_d], n)?;

    // Mark run starts, exclusive-scan → group ids + unique count.
    let mut seg_start = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.dedup_seg_start)
            .arg(&n_u64)
            .arg(&perm)
            .arg(k0_d)
            .arg(k1_d)
            .arg(k2_d)
            .arg(&mut seg_start)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let (excl, total) = excl_scan(be, stream, &seg_start, n)?;
    let m = total as usize;
    let mut out_k0 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k1 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k2 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_mult = stream.alloc_zeros::<u64>(m.max(1))?;
    unsafe {
        stream
            .launch_builder(&be.dedup_emit)
            .arg(&n_u64)
            .arg(&perm)
            .arg(k0_d)
            .arg(k1_d)
            .arg(k2_d)
            .arg(&excl)
            .arg(&seg_start)
            .arg(&mut out_k0)
            .arg(&mut out_k1)
            .arg(&mut out_k2)
            .arg(&mut out_mult)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok((out_k0, out_k1, out_k2, out_mult, m))
}

/// Dual-multiplicity dedup core (MUL: mu_lo/mu_hi; DVRM: mu_q/mu_r). Same 3-word key as
/// `dedup3_core`, plus a per-op selector bit routing each op's count into `m0` (sel=0) or
/// `m1` (sel=1). Returns unique `(k0,k1,k2,m0,m1)` device buffers + unique count.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn dedup3_core2(
    be: &Backend,
    stream: &Arc<CudaStream>,
    k0_d: &CudaSlice<u64>,
    k1_d: &CudaSlice<u64>,
    k2_d: &CudaSlice<u64>,
    sel_d: &CudaSlice<u32>,
    n: usize,
) -> Result<(
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    usize,
)> {
    let perm = radix_sort_perm(be, stream, &[k0_d, k1_d, k2_d], n)?;
    let mut seg_start = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.dedup_seg_start)
            .arg(&n_u64)
            .arg(&perm)
            .arg(k0_d)
            .arg(k1_d)
            .arg(k2_d)
            .arg(&mut seg_start)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let (excl, total) = excl_scan(be, stream, &seg_start, n)?;
    let m = total as usize;
    let mut out_k0 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k1 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k2 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_m0 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_m1 = stream.alloc_zeros::<u64>(m.max(1))?;
    unsafe {
        stream
            .launch_builder(&be.dedup_emit2)
            .arg(&n_u64)
            .arg(&perm)
            .arg(k0_d)
            .arg(k1_d)
            .arg(k2_d)
            .arg(sel_d)
            .arg(&excl)
            .arg(&seg_start)
            .arg(&mut out_k0)
            .arg(&mut out_k1)
            .arg(&mut out_k2)
            .arg(&mut out_m0)
            .arg(&mut out_m1)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok((out_k0, out_k1, out_k2, out_m0, out_m1, m))
}

/// 4-key dedup core (BRANCH: pc/offset/register/jalr). Sort by all 4 words → mark runs →
/// scan → emit unique `(k0,k1,k2,k3,mult)` + count.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn dedup4_core(
    be: &Backend,
    stream: &Arc<CudaStream>,
    k0_d: &CudaSlice<u64>,
    k1_d: &CudaSlice<u64>,
    k2_d: &CudaSlice<u64>,
    k3_d: &CudaSlice<u64>,
    n: usize,
) -> Result<(
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    CudaSlice<u64>,
    usize,
)> {
    let perm = radix_sort_perm(be, stream, &[k0_d, k1_d, k2_d, k3_d], n)?;
    let mut seg_start = stream.alloc_zeros::<u32>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.dedup_seg_start4)
            .arg(&n_u64)
            .arg(&perm)
            .arg(k0_d)
            .arg(k1_d)
            .arg(k2_d)
            .arg(k3_d)
            .arg(&mut seg_start)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let (excl, total) = excl_scan(be, stream, &seg_start, n)?;
    let m = total as usize;
    let mut out_k0 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k1 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k2 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_k3 = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut out_mult = stream.alloc_zeros::<u64>(m.max(1))?;
    unsafe {
        stream
            .launch_builder(&be.dedup_emit4)
            .arg(&n_u64)
            .arg(&perm)
            .arg(k0_d)
            .arg(k1_d)
            .arg(k2_d)
            .arg(k3_d)
            .arg(&excl)
            .arg(&seg_start)
            .arg(&mut out_k0)
            .arg(&mut out_k1)
            .arg(&mut out_k2)
            .arg(&mut out_k3)
            .arg(&mut out_mult)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    Ok((out_k0, out_k1, out_k2, out_k3, out_mult, m))
}

/// Memory walk + MEMW-row gather: run the byte-access walk (sort by address + predecessor
/// link) then scatter each access's `(old_ts, old_value)` into its op's MEMW row via the
/// `(op_row, byte_off)` mapping. Returns `(old_ts_per_op, old_value_per_op)`, each
/// `num_ops * 8` (positions beyond an op's width stay 0). The caller applies the LOAD
/// old_value=own override and the aligned/general classification. Host-in/host-out oracle.
#[allow(clippy::too_many_arguments)]
pub fn gpu_mem_walk_memw(
    addr: &[u64],
    ts: &[u64],
    value: &[u64],
    init_value: &[u64],
    op_row: &[u64],
    byte_off: &[u32],
    num_ops: usize,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = addr.len();
    if n == 0 {
        return Ok((vec![0u64; num_ops * 8], vec![0u64; num_ops * 8]));
    }
    let addr_d = stream.clone_htod(addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let init_d = stream.clone_htod(init_value)?;
    let op_row_d = stream.clone_htod(op_row)?;
    let byte_off_d = stream.clone_htod(byte_off)?;

    let perm = radix_sort_perm(be, &stream, &[&addr_d], n)?;
    let mut old_value = stream.alloc_zeros::<u64>(n)?;
    let mut old_ts = stream.alloc_zeros::<u64>(n)?;
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.mem_link)
            .arg(&perm)
            .arg(&addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&init_d)
            .arg(&n_u64)
            .arg(&mut old_value)
            .arg(&mut old_ts)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let mut out_ts = stream.alloc_zeros::<u64>(num_ops * 8)?;
    let mut out_val = stream.alloc_zeros::<u64>(num_ops * 8)?;
    unsafe {
        stream
            .launch_builder(&be.memw_gather)
            .arg(&n_u64)
            .arg(&old_ts)
            .arg(&old_value)
            .arg(&op_row_d)
            .arg(&byte_off_d)
            .arg(&mut out_ts)
            .arg(&mut out_val)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let ots = stream.clone_dtoh(&out_ts)?;
    let ov = stream.clone_dtoh(&out_val)?;
    stream.synchronize()?;
    Ok((ots, ov))
}

/// Full MEMW table assembly for LOAD/STORE ops: walk (sort+link) → per-op gather → classify
/// aligned/general → compact each bucket → pack MEMW_A (stride 12) / MEMW (stride 19) rows,
/// all on device. Returns `(packed_aligned, n_aligned, packed_general, n_general)`. Per-op
/// inputs (length num_ops): `op_base`, `op_ts`, `op_is_read` (1=LOAD), `op_width`,
/// `op_signed` (mem_signed), `op_value` (rvd for LOAD / rv2 for STORE). Byte-access inputs
/// (length n_acc) feed the walk: `addr`, `ts`, `value`, `init_value`, `op_row`, `byte_off`.
/// Unconstrained positions [width,8) of old_ts/old_value are 0 (valid; see kernel comment).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gpu_build_memw_ls(
    addr: &[u64],
    ts: &[u64],
    value: &[u64],
    init_value: &[u64],
    op_row: &[u64],
    byte_off: &[u32],
    op_base: &[u64],
    op_ts: &[u64],
    op_is_read: &[u32],
    op_width: &[u32],
    op_signed: &[u32],
    op_value: &[u64],
) -> Result<(Vec<u64>, usize, Vec<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = addr.len();
    let num_ops = op_base.len();
    if num_ops == 0 {
        return Ok((vec![], 0, vec![], 0));
    }
    // --- walk over the byte-accesses ---
    let addr_d = stream.clone_htod(addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let init_d = stream.clone_htod(init_value)?;
    let op_row_d = stream.clone_htod(op_row)?;
    let byte_off_d = stream.clone_htod(byte_off)?;
    let n_u64 = n as u64;
    let perm = radix_sort_perm(be, &stream, &[&addr_d], n)?;
    let mut old_value = stream.alloc_zeros::<u64>(n)?;
    let mut old_ts = stream.alloc_zeros::<u64>(n)?;
    unsafe {
        stream
            .launch_builder(&be.mem_link)
            .arg(&perm).arg(&addr_d).arg(&ts_d).arg(&value_d).arg(&init_d).arg(&n_u64)
            .arg(&mut old_value).arg(&mut old_ts)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    // --- gather per-op old_ts[8]/old_value[8] (resident) ---
    let mut g_ts = stream.alloc_zeros::<u64>(num_ops * 8)?;
    let mut g_val = stream.alloc_zeros::<u64>(num_ops * 8)?;
    unsafe {
        stream
            .launch_builder(&be.memw_gather)
            .arg(&n_u64).arg(&old_ts).arg(&old_value).arg(&op_row_d).arg(&byte_off_d)
            .arg(&mut g_ts).arg(&mut g_val)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    // --- classify + compact ---
    let base_d = stream.clone_htod(op_base)?;
    let opts_d = stream.clone_htod(op_ts)?;
    let isread_d = stream.clone_htod(op_is_read)?;
    let width_d = stream.clone_htod(op_width)?;
    let signed_d = stream.clone_htod(op_signed)?;
    let vword_d = stream.clone_htod(op_value)?;
    let no = num_ops as u64;
    let mut fa = stream.alloc_zeros::<u32>(num_ops)?;
    let mut fg = stream.alloc_zeros::<u32>(num_ops)?;
    unsafe {
        stream
            .launch_builder(&be.memw_classify)
            .arg(&no).arg(&base_d).arg(&width_d).arg(&g_ts).arg(&mut fa).arg(&mut fg)
            .launch(LaunchConfig::for_num_elems(num_ops as u32))?;
    }
    let (excl_a, n_a) = excl_scan(be, &stream, &fa, num_ops)?;
    let (excl_g, n_g) = excl_scan(be, &stream, &fg, num_ops)?;
    let (na, ng) = (n_a as usize, n_g as usize);
    let mut out_a = stream.alloc_zeros::<u64>(na.max(1) * 12)?;
    let mut out_g = stream.alloc_zeros::<u64>(ng.max(1) * 19)?;
    unsafe {
        stream
            .launch_builder(&be.memw_pack)
            .arg(&no).arg(&base_d).arg(&opts_d).arg(&isread_d).arg(&width_d).arg(&signed_d)
            .arg(&vword_d).arg(&g_ts).arg(&g_val).arg(&fa).arg(&excl_a).arg(&excl_g)
            .arg(&mut out_a).arg(&mut out_g)
            .launch(LaunchConfig::for_num_elems(num_ops as u32))?;
    }
    let pa = stream.clone_dtoh(&out_a)?;
    let pg = stream.clone_dtoh(&out_g)?;
    stream.synchronize()?;
    Ok((pa, na, pg, ng))
}

/// P2-memory: FULLY-RESIDENT MEMW (LOAD/STORE) table build. Emits the byte-accesses + op metadata on
/// device from the resident cpu_ops, computes `init_value` via the on-device image lookup, then walks
/// (radix sort + link) → gathers → classifies → packs MEMW_A(stride 12)/MEMW(stride 19) — all on one
/// stream with NO per-access upload (only the cpu_op SoA + the sorted image are uploaded). Returns
/// `(packed_aligned, n_aligned, packed_general, n_general)`. `img_addr` ascending. Mirrors
/// `gpu_build_memw_ls` but reads the resident emitted accesses instead of host-collected ones.
#[allow(clippy::type_complexity)]
pub fn gpu_build_memw_ls_resident(
    packed: &[u64],
    res: &[u64],
    rvd: &[u64],
    rv2: &[u64],
    img_addr: &[u64],
    img_val: &[u64],
) -> Result<(Vec<u64>, usize, Vec<u64>, usize)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_ops_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let res_d = stream.clone_htod(res)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let rv2_d = stream.clone_htod(rv2)?;

    // --- emit byte-accesses + op metadata (device) ---
    let mut ls_flag = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut byte_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.memacc_counts)
                .arg(&n_ops_u64)
                .arg(&pk_d)
                .arg(&mut ls_flag)
                .arg(&mut byte_cnt)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (op_off, num_ops_u64) = excl_scan(be, &stream, &ls_flag, n.max(1))?;
    let (byte_base, total_u64) = excl_scan(be, &stream, &byte_cnt, n.max(1))?;
    let num_ops = num_ops_u64 as usize;
    let total = total_u64 as usize;
    if num_ops == 0 {
        return Ok((vec![], 0, vec![], 0));
    }
    let mut base_d = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut opts_d = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut isread_d = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut width_d = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut signed_d = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut vword_d = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut addr_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut ts_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut value_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut op_row_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut byte_off_d = stream.alloc_zeros::<u32>(total.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.memacc_emit)
                .arg(&n_ops_u64)
                .arg(&pk_d)
                .arg(&res_d)
                .arg(&rvd_d)
                .arg(&rv2_d)
                .arg(&op_off)
                .arg(&byte_base)
                .arg(&mut base_d)
                .arg(&mut opts_d)
                .arg(&mut isread_d)
                .arg(&mut width_d)
                .arg(&mut signed_d)
                .arg(&mut vword_d)
                .arg(&mut addr_d)
                .arg(&mut ts_d)
                .arg(&mut value_d)
                .arg(&mut op_row_d)
                .arg(&mut byte_off_d)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // --- init_value via on-device image lookup (no per-access host prep) ---
    let ia_d = stream.clone_htod(img_addr)?;
    let iv_d = stream.clone_htod(img_val)?;
    let init_d = image_lookup_dev(be, &stream, &addr_d, &ia_d, &iv_d, total, img_addr.len())?;

    // --- walk (radix sort of a permutation by addr + predecessor link) ---
    let n_u64 = total as u64;
    let perm = radix_sort_perm(be, &stream, &[&addr_d], total)?;
    let mut old_value = stream.alloc_zeros::<u64>(total)?;
    let mut old_ts = stream.alloc_zeros::<u64>(total)?;
    unsafe {
        stream
            .launch_builder(&be.mem_link)
            .arg(&perm)
            .arg(&addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&init_d)
            .arg(&n_u64)
            .arg(&mut old_value)
            .arg(&mut old_ts)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }

    // --- gather per-op old_ts[8]/old_value[8] ---
    let mut g_ts = stream.alloc_zeros::<u64>(num_ops * 8)?;
    let mut g_val = stream.alloc_zeros::<u64>(num_ops * 8)?;
    unsafe {
        stream
            .launch_builder(&be.memw_gather)
            .arg(&n_u64)
            .arg(&old_ts)
            .arg(&old_value)
            .arg(&op_row_d)
            .arg(&byte_off_d)
            .arg(&mut g_ts)
            .arg(&mut g_val)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }

    // --- classify aligned/general + compact + pack ---
    let no = num_ops as u64;
    let mut fa = stream.alloc_zeros::<u32>(num_ops)?;
    let mut fg = stream.alloc_zeros::<u32>(num_ops)?;
    unsafe {
        stream
            .launch_builder(&be.memw_classify)
            .arg(&no)
            .arg(&base_d)
            .arg(&width_d)
            .arg(&g_ts)
            .arg(&mut fa)
            .arg(&mut fg)
            .launch(LaunchConfig::for_num_elems(num_ops as u32))?;
    }
    let (excl_a, n_a) = excl_scan(be, &stream, &fa, num_ops)?;
    let (excl_g, n_g) = excl_scan(be, &stream, &fg, num_ops)?;
    let (na, ng) = (n_a as usize, n_g as usize);
    let mut out_a = stream.alloc_zeros::<u64>(na.max(1) * 12)?;
    let mut out_g = stream.alloc_zeros::<u64>(ng.max(1) * 19)?;
    unsafe {
        stream
            .launch_builder(&be.memw_pack)
            .arg(&no)
            .arg(&base_d)
            .arg(&opts_d)
            .arg(&isread_d)
            .arg(&width_d)
            .arg(&signed_d)
            .arg(&vword_d)
            .arg(&g_ts)
            .arg(&g_val)
            .arg(&fa)
            .arg(&excl_a)
            .arg(&excl_g)
            .arg(&mut out_a)
            .arg(&mut out_g)
            .launch(LaunchConfig::for_num_elems(num_ops as u32))?;
    }
    let pa = stream.clone_dtoh(&out_a)?;
    let pg = stream.clone_dtoh(&out_g)?;
    stream.synchronize()?;
    Ok((pa, na, pg, ng))
}

/// memw→lt pair generation on device (LT-resident-table): from the packed MEMW_A (`pa`, `na` rows,
/// stride 12) + MEMW general (`pg`, `ng` rows, stride 19) tables, produce the timestamp-ordering LT
/// operands (lhs=old_timestamp, rhs=timestamp) — the device analog of `collect_lt_from_memw` /
/// `collect_lt_from_memw_aligned`. Returns `(lhs, rhs)` (host), MULTISET-equal to the CPU collectors
/// (the LT bus is order-free). Aligned pairs occupy `[0, na)`, general pairs `[na, na+Σwidth)`.
pub fn gpu_memw_lt_pairs(
    pa: &[u64],
    na: usize,
    pg: &[u64],
    ng: usize,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let pa_d = stream.clone_htod(pa)?;
    let pg_d = stream.clone_htod(pg)?;
    let mut widths = stream.alloc_zeros::<u32>(ng.max(1))?;
    if ng > 0 {
        let ng_u64 = ng as u64;
        unsafe {
            stream
                .launch_builder(&be.memw_lt_widths)
                .arg(&ng_u64)
                .arg(&pg_d)
                .arg(&mut widths)
                .launch(LaunchConfig::for_num_elems(ng as u32))?;
        }
    }
    let (excl_w, total_g) = excl_scan(be, &stream, &widths, ng.max(1))?;
    let total = na + total_g as usize;
    let mut lhs = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut rhs = stream.alloc_zeros::<u64>(total.max(1))?;
    if na > 0 {
        let na_u64 = na as u64;
        unsafe {
            stream
                .launch_builder(&be.memw_lt_emit_aligned)
                .arg(&na_u64)
                .arg(&pa_d)
                .arg(&mut lhs)
                .arg(&mut rhs)
                .launch(LaunchConfig::for_num_elems(na as u32))?;
        }
    }
    if ng > 0 {
        let ng_u64 = ng as u64;
        let na_u64 = na as u64;
        unsafe {
            stream
                .launch_builder(&be.memw_lt_emit_general)
                .arg(&ng_u64)
                .arg(&pg_d)
                .arg(&excl_w)
                .arg(&na_u64)
                .arg(&mut lhs)
                .arg(&mut rhs)
                .launch(LaunchConfig::for_num_elems(ng as u32))?;
        }
    }
    let h_lhs = stream.clone_dtoh(&lhs)?;
    let h_rhs = stream.clone_dtoh(&rhs)?;
    stream.synchronize()?;
    Ok((h_lhs, h_rhs))
}

/// P1-ecall: `gpu_build_memw_ls_resident` variant that INTERLEAVES the (host-captured) ecall memory
/// byte-accesses at their op's timeline position, so regular LOAD/STORE MEMW_A/MEMW rows get correct
/// old_ts/old_value across the ecall interleaving. Ecall bytes are non-emitting (routed to a DUMP row
/// `num_ops`, ignored by classify/pack — their MEMW rows are produced on CPU, Option Z). Only the tiny
/// ecall arrays are uploaded on top of the resident inputs. Returns the SAME `(MEMW_A packed, n_a,
/// MEMW packed, n_g)` as the base builder — for the regular LOAD/STORE ops only.
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_memw_ls_resident_ecall(
    packed: &[u64],
    res: &[u64],
    rvd: &[u64],
    rv2: &[u64],
    img_addr: &[u64],
    img_val: &[u64],
    ecall_op_index: &[u32],
    ecall_addr: &[u64],
    ecall_ts: &[u64],
    ecall_val: &[u64],
) -> Result<(
    Vec<u64>,
    usize,
    Vec<u64>,
    usize,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    // Option A2: per-ecall-byte resolved old_ts / old_value (parallel to the flat ecall_* inputs),
    // so the CPU can assemble the ecall MEMW rows post-walk without a memory_state replay.
    Vec<u64>,
    Vec<u64>,
)> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = packed.len();
    let n_ops_u64 = n as u64;
    let pk_d = stream.clone_htod(packed)?;
    let res_d = stream.clone_htod(res)?;
    let rvd_d = stream.clone_htod(rvd)?;
    let rv2_d = stream.clone_htod(rv2)?;

    // Per-op ecall byte counts + offsets (from the flat ecall byte arrays, grouped by op).
    let m = ecall_addr.len();
    let eidx_d = stream.clone_htod(ecall_op_index)?;
    let eaddr_d = stream.clone_htod(ecall_addr)?;
    let ets_d = stream.clone_htod(ecall_ts)?;
    let eval_d = stream.clone_htod(ecall_val)?;
    // Option A2: combined-stream position of each flat ecall byte (filled by memacc_emit_ecall,
    // read back by ecall_oldstate_gather after the walk). Sized `m` (number of ecall bytes).
    let mut ecall_pos_d = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut ecall_byte_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if m > 0 {
        unsafe {
            stream
                .launch_builder(&be.mem_ecall_byte_counts)
                .arg(&(m as u64))
                .arg(&eidx_d)
                .arg(&mut ecall_byte_cnt)
                .launch(LaunchConfig::for_num_elems(m as u32))?;
        }
    }
    let (ecall_op_off, _tot_e) = excl_scan(be, &stream, &ecall_byte_cnt, n.max(1))?;

    // --- emit byte-accesses + op metadata (device), reserving ecall byte slots per op ---
    let mut ls_flag = stream.alloc_zeros::<u32>(n.max(1))?;
    let mut byte_cnt = stream.alloc_zeros::<u32>(n.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.memacc_counts_ecall)
                .arg(&n_ops_u64)
                .arg(&pk_d)
                .arg(&ecall_byte_cnt)
                .arg(&mut ls_flag)
                .arg(&mut byte_cnt)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let (op_off, num_ops_u64) = excl_scan(be, &stream, &ls_flag, n.max(1))?;
    let (byte_base, total_u64) = excl_scan(be, &stream, &byte_cnt, n.max(1))?;
    let num_ops = num_ops_u64 as usize;
    let total = total_u64 as usize;
    if num_ops == 0 {
        return Ok((vec![], 0, vec![], 0, vec![], vec![], vec![], vec![], vec![]));
    }
    let mut base_d = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut opts_d = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut isread_d = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut width_d = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut signed_d = stream.alloc_zeros::<u32>(num_ops.max(1))?;
    let mut vword_d = stream.alloc_zeros::<u64>(num_ops.max(1))?;
    let mut addr_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut ts_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut value_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut op_row_d = stream.alloc_zeros::<u64>(total.max(1))?;
    let mut byte_off_d = stream.alloc_zeros::<u32>(total.max(1))?;
    if n > 0 {
        unsafe {
            stream
                .launch_builder(&be.memacc_emit)
                .arg(&n_ops_u64)
                .arg(&pk_d)
                .arg(&res_d)
                .arg(&rvd_d)
                .arg(&rv2_d)
                .arg(&op_off)
                .arg(&byte_base)
                .arg(&mut base_d)
                .arg(&mut opts_d)
                .arg(&mut isread_d)
                .arg(&mut width_d)
                .arg(&mut signed_d)
                .arg(&mut vword_d)
                .arg(&mut addr_d)
                .arg(&mut ts_d)
                .arg(&mut value_d)
                .arg(&mut op_row_d)
                .arg(&mut byte_off_d)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        // Interleave the ecall memory bytes (non-emitting DUMP rows).
        let num_ops_dump = num_ops as u64;
        unsafe {
            stream
                .launch_builder(&be.memacc_emit_ecall)
                .arg(&n_ops_u64)
                .arg(&pk_d)
                .arg(&ecall_byte_cnt)
                .arg(&ecall_op_off)
                .arg(&byte_base)
                .arg(&eaddr_d)
                .arg(&ets_d)
                .arg(&eval_d)
                .arg(&num_ops_dump)
                .arg(&mut addr_d)
                .arg(&mut ts_d)
                .arg(&mut value_d)
                .arg(&mut op_row_d)
                .arg(&mut byte_off_d)
                .arg(&mut ecall_pos_d)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // --- init_value via on-device image lookup ---
    let ia_d = stream.clone_htod(img_addr)?;
    let iv_d = stream.clone_htod(img_val)?;
    let init_d = image_lookup_dev(be, &stream, &addr_d, &ia_d, &iv_d, total, img_addr.len())?;

    // --- walk (radix sort by addr + predecessor link) over the combined byte stream ---
    let n_u64 = total as u64;
    let perm = radix_sort_perm(be, &stream, &[&addr_d], total)?;
    let mut old_value = stream.alloc_zeros::<u64>(total)?;
    let mut old_ts = stream.alloc_zeros::<u64>(total)?;
    unsafe {
        stream
            .launch_builder(&be.mem_link)
            .arg(&perm)
            .arg(&addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&init_d)
            .arg(&n_u64)
            .arg(&mut old_value)
            .arg(&mut old_ts)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }

    // --- Option A2: read back each ecall byte's resolved old_ts/old_value (parallel to the flat
    //     ecall_* inputs) so the CPU can assemble the ecall MEMW rows without a memory_state replay ---
    let mut ecall_old_ts_d = stream.alloc_zeros::<u64>(m.max(1))?;
    let mut ecall_old_val_d = stream.alloc_zeros::<u64>(m.max(1))?;
    if m > 0 {
        unsafe {
            stream
                .launch_builder(&be.ecall_oldstate_gather)
                .arg(&(m as u64))
                .arg(&ecall_pos_d)
                .arg(&old_ts)
                .arg(&old_value)
                .arg(&mut ecall_old_ts_d)
                .arg(&mut ecall_old_val_d)
                .launch(LaunchConfig::for_num_elems(m as u32))?;
        }
    }
    let ecall_old_ts = if m == 0 {
        Vec::new()
    } else {
        stream.clone_dtoh(&ecall_old_ts_d)?
    };
    let ecall_old_val = if m == 0 {
        Vec::new()
    } else {
        stream.clone_dtoh(&ecall_old_val_d)?
    };

    // --- gather per-op old_ts[8]/old_value[8]; DUMP row `num_ops` absorbs the ecall bytes ---
    let mut g_ts = stream.alloc_zeros::<u64>((num_ops + 1) * 8)?;
    let mut g_val = stream.alloc_zeros::<u64>((num_ops + 1) * 8)?;
    unsafe {
        stream
            .launch_builder(&be.memw_gather)
            .arg(&n_u64)
            .arg(&old_ts)
            .arg(&old_value)
            .arg(&op_row_d)
            .arg(&byte_off_d)
            .arg(&mut g_ts)
            .arg(&mut g_val)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }

    // --- FINAL memory snapshot: last access per address (value, ts) after the full replay (regular +
    //     interleaved ecall writes are all in the sorted stream) — feeds PAGE-FINI + ARE_BYTES ---
    let mut fflag = stream.alloc_zeros::<u32>(total)?;
    unsafe {
        stream
            .launch_builder(&be.mem_final_flag)
            .arg(&perm)
            .arg(&addr_d)
            .arg(&n_u64)
            .arg(&mut fflag)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let (fexcl, nfin_u64) = excl_scan(be, &stream, &fflag, total)?;
    let nfin = nfin_u64 as usize;
    let mut snap_addr = stream.alloc_zeros::<u64>(nfin.max(1))?;
    let mut snap_val = stream.alloc_zeros::<u64>(nfin.max(1))?;
    let mut snap_ts = stream.alloc_zeros::<u64>(nfin.max(1))?;
    unsafe {
        stream
            .launch_builder(&be.mem_final_gather)
            .arg(&perm)
            .arg(&addr_d)
            .arg(&ts_d)
            .arg(&value_d)
            .arg(&fflag)
            .arg(&fexcl)
            .arg(&n_u64)
            .arg(&mut snap_addr)
            .arg(&mut snap_val)
            .arg(&mut snap_ts)
            .launch(LaunchConfig::for_num_elems(total as u32))?;
    }
    let sa = if nfin == 0 { Vec::new() } else { stream.clone_dtoh(&snap_addr)? };
    let sv = if nfin == 0 { Vec::new() } else { stream.clone_dtoh(&snap_val)? };
    let st = if nfin == 0 { Vec::new() } else { stream.clone_dtoh(&snap_ts)? };

    // --- classify aligned/general + compact + pack (regular ops only; dump row ignored) ---
    let no = num_ops as u64;
    let mut fa = stream.alloc_zeros::<u32>(num_ops)?;
    let mut fg = stream.alloc_zeros::<u32>(num_ops)?;
    unsafe {
        stream
            .launch_builder(&be.memw_classify)
            .arg(&no)
            .arg(&base_d)
            .arg(&width_d)
            .arg(&g_ts)
            .arg(&mut fa)
            .arg(&mut fg)
            .launch(LaunchConfig::for_num_elems(num_ops as u32))?;
    }
    let (excl_a, n_a) = excl_scan(be, &stream, &fa, num_ops)?;
    let (excl_g, n_g) = excl_scan(be, &stream, &fg, num_ops)?;
    let (na, ng) = (n_a as usize, n_g as usize);
    let mut out_a = stream.alloc_zeros::<u64>(na.max(1) * 12)?;
    let mut out_g = stream.alloc_zeros::<u64>(ng.max(1) * 19)?;
    unsafe {
        stream
            .launch_builder(&be.memw_pack)
            .arg(&no)
            .arg(&base_d)
            .arg(&opts_d)
            .arg(&isread_d)
            .arg(&width_d)
            .arg(&signed_d)
            .arg(&vword_d)
            .arg(&g_ts)
            .arg(&g_val)
            .arg(&fa)
            .arg(&excl_a)
            .arg(&excl_g)
            .arg(&mut out_a)
            .arg(&mut out_g)
            .launch(LaunchConfig::for_num_elems(num_ops as u32))?;
    }
    let pa = stream.clone_dtoh(&out_a)?;
    let pg = stream.clone_dtoh(&out_g)?;
    stream.synchronize()?;
    // (pa, na, pg, ng, snapshot addr/value/ts, ecall-byte old_ts/old_value)
    Ok((pa, na, pg, ng, sa, sv, st, ecall_old_ts, ecall_old_val))
}
