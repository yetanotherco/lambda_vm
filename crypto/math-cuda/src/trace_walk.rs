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

/// Device exclusive prefix scan of a 0/1 `flag` array of length `n`. Returns the
/// per-element exclusive prefix (device) and the grand total (host). Two-level
/// (per-block totals → spine scan → per-block write); see `trace_walk.cu`.
fn excl_scan(
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
    assert_eq!(init_value.len(), nbins as usize);

    // Upload the access SoA once; shared by walk, route, fill, and gather.
    let reg_addr_d = stream.clone_htod(reg_addr)?;
    let ts_d = stream.clone_htod(ts)?;
    let value_d = stream.clone_htod(value)?;
    let is_read_d = stream.clone_htod(is_read)?;
    let emits_row_d = stream.clone_htod(emits_row)?;
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
