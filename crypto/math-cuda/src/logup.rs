//! GPU LogUp aux build kernels.
//!
//! Two stages, mirroring `stark::logup_gpu`:
//!   1. `logup_fingerprints_dev`: one ext3 fingerprint per (interaction, row).
//!   2. `logup_term_columns`: fingerprints -> batch inverse -> per-output-column
//!      signed-multiplicity combine, producing the committed + virtual term
//!      columns.
//!
//! The descriptor is passed as plain array slices ([`LogupDescriptor`]) so this
//! crate stays independent of the stark types.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::inverse::batch_inverse_ext3_dev;

// Must match LOGUP_BLK in kernels/logup.cu: the block scan kernel assumes
// exactly this many threads per block for its shared-memory array.
const BLOCK_SIZE: u32 = 256;

/// Flat LogUp descriptor for one table (CSR arrays, canonical Goldilocks). Built
/// by `stark::logup_gpu::build_fingerprint_descriptor`.
pub struct LogupDescriptor<'a> {
    pub num_interactions: usize,
    // fingerprint
    pub bus_ids: &'a [u64],
    pub elem_offsets: &'a [u32],
    pub elem_alpha_idx: &'a [u32],
    pub elem_const: &'a [u64],
    pub term_offsets: &'a [u32],
    pub term_coef: &'a [u64],
    pub term_col: &'a [u32],
    // term combine
    pub num_out_cols: usize,
    pub out_col_offsets: &'a [u32],
    pub out_col_interactions: &'a [u32],
    pub mult_const: &'a [u64],
    pub mult_term_offsets: &'a [u32],
    pub mult_term_coef: &'a [u64],
    pub mult_term_col: &'a [u32],
}

fn cfg(total: usize) -> Result<LaunchConfig> {
    // See `batch_inverse_ext3_dev` for the rationale: a u32 grid_dim is
    // truncated past u32::MAX / BLOCK_SIZE, which would silently launch too
    // few blocks and leave a tail of the (uninitialized) output unwritten.
    // Runtime Err, not debug_assert, so release builds also route to the
    // caller's CPU fallback.
    if total > u32::MAX as usize / BLOCK_SIZE as usize {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ));
    }
    Ok(LaunchConfig {
        grid_dim: ((total as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Fingerprint kernel over a device-resident main trace. Returns the ext3 fp
/// buffer (`num_interactions * num_rows * 3`, layout `[(k*num_rows+row)*3+limb]`).
fn fingerprints_into_dev(
    main_dev: &CudaSlice<u64>,
    num_rows: usize,
    d: &LogupDescriptor,
    alpha_powers: &[u64],
    z: [u64; 3],
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    let total = d.num_interactions * num_rows;
    let mut out = unsafe { stream.alloc::<u64>(total * 3) }?;
    if total == 0 {
        return Ok(out);
    }
    let be = backend()?;
    let bus_ids = stream.clone_htod(d.bus_ids)?;
    let elem_offsets = stream.clone_htod(d.elem_offsets)?;
    let elem_alpha_idx = stream.clone_htod(d.elem_alpha_idx)?;
    let elem_const = stream.clone_htod(d.elem_const)?;
    let term_offsets = stream.clone_htod(d.term_offsets)?;
    let term_coef = stream.clone_htod(d.term_coef)?;
    let term_col = stream.clone_htod(d.term_col)?;
    let alpha = stream.clone_htod(alpha_powers)?;
    let num_rows_u32 = num_rows as u32;
    let num_int_u32 = d.num_interactions as u32;
    let (z0, z1, z2) = (z[0], z[1], z[2]);
    unsafe {
        stream
            .launch_builder(&be.logup_fingerprint_ext3)
            .arg(main_dev)
            .arg(&num_rows_u32)
            .arg(&num_int_u32)
            .arg(&bus_ids)
            .arg(&elem_offsets)
            .arg(&elem_alpha_idx)
            .arg(&elem_const)
            .arg(&term_offsets)
            .arg(&term_coef)
            .arg(&term_col)
            .arg(&alpha)
            .arg(&z0)
            .arg(&z1)
            .arg(&z2)
            .arg(&mut out)
            .launch(cfg(total)?)?;
    }
    Ok(out)
}

/// Compute fingerprints from a host main trace (column-major, `num_cols*num_rows`),
/// returning the resident ext3 buffer. The stream is synchronised before return.
pub fn logup_fingerprints_dev(
    main_cols: &[u64],
    num_rows: usize,
    d: &LogupDescriptor,
    alpha_powers: &[u64],
    z: [u64; 3],
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    let _nvtx = crate::nvtx::Range::fmt(|| format!("logup_fingerprints[rows={num_rows}]"));
    let main_dev = stream.clone_htod(main_cols)?;
    let out = fingerprints_into_dev(&main_dev, num_rows, d, alpha_powers, z, stream)?;
    stream.synchronize()?;
    Ok(out)
}

/// Full term-column pipeline: fingerprints -> batch inverse -> term combine.
/// Returns the host term columns (`num_out_cols * num_rows * 3`, ext3
/// interleaved, layout `[(col*num_rows+row)*3+limb]`).
pub fn logup_term_columns(
    main_cols: &[u64],
    num_rows: usize,
    d: &LogupDescriptor,
    alpha_powers: &[u64],
    z: [u64; 3],
) -> Result<Vec<u64>> {
    let _nvtx = crate::nvtx::Range::fmt(|| format!("logup_terms[rows={num_rows}]"));
    let be = backend()?;
    let stream = be.next_stream();
    let timing = std::env::var_os("LAMBDA_VM_LOGUP_TIMING").is_some();
    let t0 = std::time::Instant::now();
    let main_dev = { stream.clone_htod(main_cols)? };
    if timing {
        stream.synchronize()?;
    }
    let t1 = std::time::Instant::now();

    let fp = { fingerprints_into_dev(&main_dev, num_rows, d, alpha_powers, z, &stream)? };
    let n = d.num_interactions * num_rows;
    let recip = { batch_inverse_ext3_dev(&fp, n, &stream)? };

    let total = d.num_out_cols * num_rows;
    let mut out = unsafe { stream.alloc::<u64>(total * 3) }?;
    if total == 0 {
        stream.synchronize()?;
        return Ok(Vec::new());
    }

    let (
        out_col_offsets,
        out_col_interactions,
        mult_const,
        mult_term_offsets,
        mult_term_coef,
        mult_term_col,
    ) = {
        (
            stream.clone_htod(d.out_col_offsets)?,
            stream.clone_htod(d.out_col_interactions)?,
            stream.clone_htod(d.mult_const)?,
            stream.clone_htod(d.mult_term_offsets)?,
            stream.clone_htod(d.mult_term_coef)?,
            stream.clone_htod(d.mult_term_col)?,
        )
    };
    let num_rows_u32 = num_rows as u32;
    let num_out_u32 = d.num_out_cols as u32;
    unsafe {
        stream
            .launch_builder(&be.logup_term_ext3)
            .arg(&main_dev)
            .arg(&num_rows_u32)
            .arg(&recip)
            .arg(&num_out_u32)
            .arg(&out_col_offsets)
            .arg(&out_col_interactions)
            .arg(&mult_const)
            .arg(&mult_term_offsets)
            .arg(&mult_term_coef)
            .arg(&mult_term_col)
            .arg(&mut out)
            .launch(cfg(total)?)?;
    }
    if timing {
        stream.synchronize()?;
    }
    let t2 = std::time::Instant::now();
    // Terms download (num_out_cols * num_rows * 3 u64s): async D2H through
    // the per-worker pinned slab instead of a blocking pageable copy. The
    // labelled sync keeps its host-block measurement (now covering the
    // kernels plus the DMA); the pending wait after it is instant.
    let pending =
        { crate::device::async_dtoh_via(&stream, be.pinned_staging(), &be.ctx, &out, total * 3)? };
    {
        stream.synchronize()?;
    }
    let mut host = vec![0u64; total * 3];
    pending.wait_into_u64(&mut host)?;
    let t3 = std::time::Instant::now();
    if timing {
        eprintln!(
            "LOGUP_GPU rows={} cols={} h2d_main={:?} compute={:?} d2h_terms={:?}",
            num_rows,
            main_cols.len() / num_rows,
            t1 - t0,
            t2 - t1,
            t3 - t2,
        );
    }
    Ok(host)
}

// Additive multi-block inclusive scan (mirrors inverse::scan_into_fwd, add).
fn scan_add_inplace(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    buf: &mut CudaSlice<u64>,
    n: usize,
) -> Result<()> {
    if n <= 1 {
        return Ok(());
    }
    let k = (n as u32).div_ceil(BLOCK_SIZE);
    let mut scan_out = unsafe { stream.alloc::<u64>(3 * n) }?;
    let mut block_totals = unsafe { stream.alloc::<u64>(3 * k as usize) }?;
    let n_u64 = n as u64;
    let phase = LaunchConfig {
        grid_dim: (k, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.logup_scan_block_add_ext3)
            .arg(&*buf)
            .arg(&n_u64)
            .arg(&mut scan_out)
            .arg(&mut block_totals)
            .launch(phase)?;
    }
    if k > 1 {
        scan_add_inplace(stream, be, &mut block_totals, k as usize)?;
        unsafe {
            stream
                .launch_builder(&be.logup_apply_offsets_add_ext3)
                .arg(&mut scan_out)
                .arg(&n_u64)
                .arg(&block_totals)
                .launch(phase)?;
        }
    }
    stream.memcpy_dtod(&scan_out, buf)?;
    Ok(())
}

/// The aux trace produced entirely on device: the row-major ext3 aux columns
/// resident on the GPU (fed straight to the aux LDE, no host round-trip), the
/// column count, and the host-side table contribution `L`.
#[derive(Clone)]
pub struct ResidentAux {
    /// Row-major ext3 aux columns `[row * num_aux_cols + col]` (`committed + 1`).
    pub buf: Arc<CudaSlice<u64>>,
    pub num_aux_cols: usize,
    pub num_rows: usize,
    /// LogUp table contribution (`L`), for the bus public inputs.
    pub table_contribution: [u64; 3],
}

// Debug/PartialEq/Eq compare only the host-side metadata (the device buffer is
// not comparable and never differs when the metadata matches); these exist so a
// `TraceTable` holding an optional `ResidentAux` can keep its derives.
impl std::fmt::Debug for ResidentAux {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentAux")
            .field("num_aux_cols", &self.num_aux_cols)
            .field("num_rows", &self.num_rows)
            .finish()
    }
}
impl PartialEq for ResidentAux {
    fn eq(&self, other: &Self) -> bool {
        self.num_aux_cols == other.num_aux_cols
            && self.num_rows == other.num_rows
            && self.table_contribution == other.table_contribution
    }
}
impl Eq for ResidentAux {}

/// Main trace input for the resident aux build: either a host column-major
/// buffer to upload, or an already-resident device buffer (from the R1 main
/// LDE) to read in place. The device form skips the ~3 GB main re-upload.
#[derive(Clone, Copy)]
pub enum ResidentMain<'a> {
    Host(&'a [u64]),
    Dev(&'a CudaSlice<u64>),
}

/// Full aux build on device: fingerprints → invert → term columns → accumulate
/// scan → assemble the row-major aux trace buffer, all resident. `inv_n` is
/// `1/num_rows` embedded in ext3. Requires `num_rows >= 1`. The stream is
/// synchronised before return.
#[allow(clippy::too_many_arguments)]
pub fn logup_aux_resident(
    main: ResidentMain,
    num_rows: usize,
    d: &LogupDescriptor,
    alpha_powers: &[u64],
    z: [u64; 3],
    inv_n: [u64; 3],
    stream: &Arc<CudaStream>,
) -> Result<ResidentAux> {
    let _nvtx = crate::nvtx::Range::fmt(|| format!("logup_aux[rows={num_rows}]"));
    assert!(num_rows >= 1, "logup_aux_resident requires num_rows >= 1");
    let be = backend()?;
    // Per-phase timing (env LAMBDA_VM_LOGUP_TIMING): sync between phases so wall
    // time is attributed correctly. Off = no extra syncs, production path.
    let timing = std::env::var_os("LAMBDA_VM_LOGUP_TIMING").is_some();
    let sync_if = |on: bool| -> Result<()> {
        if on {
            stream.synchronize()?;
        }
        Ok(())
    };
    let t0 = std::time::Instant::now();

    // Resident device main = zero upload; host main = one H2D. `uploaded` owns
    // the staged buffer for the function scope so `main_dev` can borrow it.
    let uploaded: Option<CudaSlice<u64>> = {
        match main {
            ResidentMain::Dev(_) => None,
            ResidentMain::Host(h) => Some(stream.clone_htod(h)?),
        }
    };
    let main_dev: &CudaSlice<u64> = match (main, &uploaded) {
        (ResidentMain::Dev(d), _) => d,
        (ResidentMain::Host(_), Some(up)) => up,
        _ => unreachable!(),
    };
    let main_len = main_dev.len();
    sync_if(timing)?;
    let t_h2d = std::time::Instant::now();

    let fp = { fingerprints_into_dev(main_dev, num_rows, d, alpha_powers, z, stream)? };
    sync_if(timing)?;
    let t_fp = std::time::Instant::now();

    let n = d.num_interactions * num_rows;
    let recip = { batch_inverse_ext3_dev(&fp, n, stream)? };
    sync_if(timing)?;
    let t_inv = std::time::Instant::now();

    // Term columns (committed + virtual), resident, layout [col][row].
    // num_out is always >= 1 (the accumulated column); num_committed = num_out - 1.
    // Runtime Err, not debug_assert: in release a zero would wrap num_out - 1
    // to usize::MAX and launch the assemble kernel with a bogus column count.
    if d.num_out_cols == 0 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ));
    }
    let num_out = d.num_out_cols;
    let mut terms = unsafe { stream.alloc::<u64>(num_out * num_rows * 3) }?;
    let (
        out_col_offsets,
        out_col_interactions,
        mult_const,
        mult_term_offsets,
        mult_term_coef,
        mult_term_col,
    ) = {
        (
            stream.clone_htod(d.out_col_offsets)?,
            stream.clone_htod(d.out_col_interactions)?,
            stream.clone_htod(d.mult_const)?,
            stream.clone_htod(d.mult_term_offsets)?,
            stream.clone_htod(d.mult_term_coef)?,
            stream.clone_htod(d.mult_term_col)?,
        )
    };
    sync_if(timing)?;
    let t_desc = std::time::Instant::now();
    let num_rows_u32 = num_rows as u32;
    let num_out_u32 = num_out as u32;
    unsafe {
        stream
            .launch_builder(&be.logup_term_ext3)
            .arg(main_dev)
            .arg(&num_rows_u32)
            .arg(&recip)
            .arg(&num_out_u32)
            .arg(&out_col_offsets)
            .arg(&out_col_interactions)
            .arg(&mult_const)
            .arg(&mult_term_offsets)
            .arg(&mult_term_coef)
            .arg(&mult_term_col)
            .arg(&mut terms)
            .launch(cfg(num_out * num_rows)?)?;
    }
    sync_if(timing)?;
    let t_term = std::time::Instant::now();

    // row_sum over all term columns → additive scan → accumulated column.
    let num_committed = num_out - 1;
    let num_aux_cols = num_committed + 1;
    let mut row_sum;
    let mut aux;
    {
        row_sum = unsafe { stream.alloc::<u64>(num_rows * 3) }?;
        unsafe {
            stream
                .launch_builder(&be.logup_row_sum_ext3)
                .arg(&terms)
                .arg(&num_out_u32)
                .arg(&num_rows_u32)
                .arg(&mut row_sum)
                .launch(cfg(num_rows)?)?;
        }
        scan_add_inplace(stream, be, &mut row_sum, num_rows)?; // row_sum now holds S
        let (i0, i1, i2) = (inv_n[0], inv_n[1], inv_n[2]);
        let mut accumulated = unsafe { stream.alloc::<u64>(num_rows * 3) }?;
        let n_u64 = num_rows as u64;
        unsafe {
            stream
                .launch_builder(&be.logup_finalize_accum_ext3)
                .arg(&row_sum)
                .arg(&n_u64)
                .arg(&i0)
                .arg(&i1)
                .arg(&i2)
                .arg(&mut accumulated)
                .launch(cfg(num_rows)?)?;
        }

        // Assemble row-major aux buffer: committed (num_out-1) cols + accumulated.
        aux = unsafe { stream.alloc::<u64>(num_aux_cols * num_rows * 3) }?;
        let num_committed_u32 = num_committed as u32;
        unsafe {
            stream
                .launch_builder(&be.logup_assemble_aux_ext3)
                .arg(&terms)
                .arg(&num_committed_u32)
                .arg(&accumulated)
                .arg(&num_rows_u32)
                .arg(&mut aux)
                .launch(cfg(num_rows)?)?;
        }
    }
    sync_if(timing)?;
    let t_accum_done = std::time::Instant::now();

    // L = table_contribution = S[n-1] (sum of all term columns, all rows).
    let l_host: Vec<u64> = { stream.clone_dtoh(&row_sum.slice((num_rows - 1) * 3..num_rows * 3))? };
    {
        stream.synchronize()?;
    }
    if timing {
        let t_end = std::time::Instant::now();
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let main_mb = (main_len * 8) as f64 / 1e6;
        eprintln!(
            "LOGUP_RESIDENT rows={} out_cols={} interactions={} main={:.0}MB | \
             h2d_main={:.2} fp={:.2} inv={:.2} desc_up={:.2} term={:.2} accum={:.2} l_dtoh={:.2} total={:.2} ms",
            num_rows,
            num_out,
            d.num_interactions,
            main_mb,
            ms(t0, t_h2d),
            ms(t_h2d, t_fp),
            ms(t_fp, t_inv),
            ms(t_inv, t_desc),
            ms(t_desc, t_term),
            ms(t_term, t_accum_done),
            ms(t_accum_done, t_end),
            ms(t0, t_end),
        );
    }
    Ok(ResidentAux {
        buf: Arc::new(aux),
        num_aux_cols,
        num_rows,
        table_contribution: [l_host[0], l_host[1], l_host[2]],
    })
}
