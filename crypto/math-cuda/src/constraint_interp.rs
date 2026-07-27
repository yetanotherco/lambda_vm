//! Host wrapper for the transition-constraint interpreter kernel
//! (`kernels/constraint_interp.cu`).
//!
//! Takes a constraint program already lowered to flat `u64` device arrays (by
//! `stark::constraint_ir::device::DeviceProgram`) plus the device-resident LDE
//! handles, uploads the program + per-proof uniforms, launches the interpreter
//! over every LDE row, and returns the per-constraint eval matrix.
//!
//! The lowering dim-splits the per-thread value scratch into a base (`u64`)
//! and an ext (`3 × u64`) slot class with liveness-reused slots, so the
//! scratch here is sized by the program's max-live-set
//! (`num_base_slots`/`num_ext_slots`), not its node count. Both buffers are
//! allocated uninitialized: the topological walk writes every slot before any
//! read.
//!
//! Layering note: this crate cannot see `stark`'s `DeviceProgram` type (stark
//! depends on math-cuda, not the reverse), so the caller flattens the program
//! into the raw `u64` slices below. The stark-side dispatch
//! (`stark::constraint_ir::gpu_interp`) owns that flattening + the TypeId gate.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::lde::{GpuLdeBase, GpuLdeExt3};

const BLOCK_DIM: u32 = 256;
/// Cap on total threads (grid × block). Each thread owns `num_base_slots` u64
/// plus `num_ext_slots` ext3 slots of global value scratch, so a fixed cap
/// bounds those buffers regardless of LDE size; threads grid-stride over the
/// remaining rows. 65536 mirrors OpenVM's quotient `TASK_SIZE`.
const MAX_THREADS: u32 = 1 << 16;

/// Evaluate every constraint of a lowered program over the device-resident LDE.
///
/// Returns the per-constraint eval matrix as raw ext3 limbs, constraint-major:
/// constraint `c`, row `r`, component `k` at `out[(c * num_rows + r) * 3 + k]`.
/// Base-rooted constraints carry their value in component 0.
///
/// Inputs (all raw limbs, matching the crate's u64 device convention):
/// - `nodes`: 2 `u64` per IR node (`op | a<<32`, then `b | res<<32`).
/// - `num_base_slots` / `num_ext_slots`: per-thread scratch sizes of the two
///   slot classes (from the lowering's liveness scan).
/// - `base_consts`: one `u64` per base constant.
/// - `ext_consts`, `rap_challenges`, `alpha_powers`: 3 `u64` per element.
/// - `table_offset`: exactly 3 `u64`.
/// - `roots`: one `u64` per constraint (`slot | ext_bit<<31`).
/// - `main`/`aux`: device-resident LDE handles; `next_step` is the LDE row
///   stride for a frame-offset step; `num_rows` is the number of LDE rows.
#[allow(clippy::too_many_arguments)]
pub fn eval_constraints_on_device(
    nodes: &[u64],
    num_nodes: usize,
    num_base_slots: usize,
    num_ext_slots: usize,
    base_consts: &[u64],
    ext_consts: &[u64],
    roots: &[u64],
    rap_challenges: &[u64],
    alpha_powers: &[u64],
    table_offset: &[u64],
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    next_step: usize,
    num_rows: usize,
) -> Result<Vec<u64>> {
    let num_roots = roots.len();
    if num_rows == 0 || num_roots == 0 || num_nodes == 0 {
        return Ok(vec![0u64; num_roots * num_rows * 3]);
    }
    debug_assert_eq!(nodes.len(), 2 * num_nodes, "2 u64 per node");
    debug_assert_eq!(table_offset.len(), 3, "table_offset is one ext3 element");

    let be = backend()?;
    let stream = be.next_stream();
    main.wait_ready_on(&stream)?;
    aux.wait_ready_on(&stream)?;

    // Upload the program + uniforms (the column data never crosses PCIe — it is
    // already resident in `main.buf` / `aux.buf`).
    let (d_nodes, d_base_consts, d_ext_consts, d_roots, d_rap, d_alpha, d_offset) = {
        (
            stream.clone_htod(nodes)?,
            stream.clone_htod(base_consts)?,
            stream.clone_htod(ext_consts)?,
            stream.clone_htod(roots)?,
            stream.clone_htod(rap_challenges)?,
            stream.clone_htod(alpha_powers)?,
            stream.clone_htod(table_offset)?,
        )
    };

    // Fixed thread count, grid-stride over rows.
    let max_grid = MAX_THREADS / BLOCK_DIM;
    let grid = (num_rows as u32).div_ceil(BLOCK_DIM).clamp(1, max_grid);
    let num_threads = (grid as usize) * (BLOCK_DIM as usize);

    // Per-thread slot scratch, uninitialized (the walk writes before reading).
    let mut d_vals_base = unsafe { stream.alloc::<u64>((num_base_slots * num_threads).max(1)) }?;
    let mut d_vals_ext = unsafe { stream.alloc::<u64>((num_ext_slots * 3 * num_threads).max(1)) }?;
    // Output: every (constraint, row) cell is written by the emit loop.
    let mut d_evals = unsafe { stream.alloc::<u64>(num_roots * num_rows * 3) }?;

    let num_nodes_u64 = num_nodes as u64;
    let num_roots_u64 = num_roots as u64;
    let main_stride = main.lde_size as u64;
    let aux_stride = aux.lde_size as u64;
    let next_step_u64 = next_step as u64;
    let num_rows_u64 = num_rows as u64;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.constraint_interp_kernel)
            .arg(&mut d_evals)
            .arg(&d_nodes)
            .arg(&num_nodes_u64)
            .arg(&d_base_consts)
            .arg(&d_ext_consts)
            .arg(&d_roots)
            .arg(&num_roots_u64)
            .arg(&d_rap)
            .arg(&d_alpha)
            .arg(&d_offset)
            .arg(main.buf.as_ref())
            .arg(&main_stride)
            .arg(aux.buf.as_ref())
            .arg(&aux_stride)
            .arg(&next_step_u64)
            .arg(&num_rows_u64)
            .arg(&mut d_vals_base)
            .arg(&mut d_vals_ext)
            .launch(cfg)?;
    }
    let out = {
        let pending = crate::device::async_dtoh_via(
            &stream,
            be.pinned_staging(),
            &be.ctx,
            &d_evals,
            d_evals.len(),
        )?;
        let mut out = vec![0u64; d_evals.len()];
        pending.wait_into_u64(&mut out)?;
        out
    };
    Ok(out)
}

/// The per-proof accumulation inputs that turn per-constraint evals into the
/// composition-poly evaluation `H(row)` (all raw limbs; see
/// [`eval_composition_on_device`]).
pub struct CompositionAccum<'a> {
    /// Transition combination coefficients β, one ext3 per constraint root
    /// (`num_roots * 3` u64).
    pub beta_trans: &'a [u64],
    /// Cyclic transition-zerofier inverse, base field (`z_len` u64), indexed
    /// `row % z_len`.
    pub z_inv: &'a [u64],
    /// Boundary constraint columns (`num_boundary` u64).
    pub b_col: &'a [u64],
    /// Boundary main/aux selector, 0 = main / 1 = aux (`num_boundary` u64).
    pub b_is_aux: &'a [u64],
    /// Boundary target values (`num_boundary * 3` u64, ext3).
    pub b_value: &'a [u64],
    /// Boundary combination coefficients β_b (`num_boundary * 3` u64, ext3).
    pub b_beta: &'a [u64],
    /// Boundary zerofier inverses, base field: one device-resident column per
    /// boundary constraint (see [`GpuBaseVec`]). D2D-copied into one flat
    /// device buffer (kernel indexing `b * num_rows + row`) — no PCIe traffic
    /// per dispatch.
    pub b_z_inv: &'a [&'a GpuBaseVec],
}

/// A base-field column resident on device, uploaded once and reused across
/// dispatches (e.g. a boundary-zerofier inverse vector, identical for every
/// table/epoch sharing a domain). The upload synchronizes its stream, so any
/// later stream may read the buffer.
pub struct GpuBaseVec {
    buf: CudaSlice<u64>,
    len: usize,
}

impl GpuBaseVec {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub fn upload_base_vec(v: &[u64]) -> Result<GpuBaseVec> {
    let be = backend()?;
    let stream = be.next_stream();
    let buf = stream.clone_htod(v)?;
    stream.synchronize()?;
    Ok(GpuBaseVec { buf, len: v.len() })
}

/// Launch the fused composition evaluation and return the device-resident
/// result plus its stream (shared body of [`eval_composition_on_device`] and
/// [`eval_composition_on_device_keep`]).
#[allow(clippy::too_many_arguments)]
fn eval_composition_launch(
    nodes: &[u64],
    num_nodes: usize,
    num_base_slots: usize,
    num_ext_slots: usize,
    base_consts: &[u64],
    ext_consts: &[u64],
    roots: &[u64],
    rap_challenges: &[u64],
    alpha_powers: &[u64],
    table_offset: &[u64],
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    next_step: usize,
    num_rows: usize,
    accum: &CompositionAccum,
) -> Result<(CudaSlice<u64>, Arc<CudaStream>)> {
    let num_roots = roots.len();
    assert!(num_rows > 0, "callers gate empty domains");
    debug_assert_eq!(nodes.len(), 2 * num_nodes, "2 u64 per node");
    debug_assert_eq!(accum.beta_trans.len(), num_roots * 3, "β per root");
    let num_boundary = accum.b_col.len();
    debug_assert_eq!(accum.b_z_inv.len(), num_boundary, "z_b_inv per boundary");
    debug_assert!(
        accum.b_z_inv.iter().all(|s| s.len() == num_rows),
        "z_b_inv slice shape"
    );
    // The kernel indexes these by `num_boundary`; a caller mismatch would be an
    // OOB device read rather than a clean panic, so pin all boundary shapes.
    debug_assert_eq!(accum.b_is_aux.len(), num_boundary, "b_is_aux per boundary");
    debug_assert_eq!(
        accum.b_value.len(),
        num_boundary * 3,
        "b_value ext3 per boundary"
    );
    debug_assert_eq!(
        accum.b_beta.len(),
        num_boundary * 3,
        "b_beta ext3 per boundary"
    );

    let be = backend()?;
    let stream = be.next_stream();
    main.wait_ready_on(&stream)?;
    aux.wait_ready_on(&stream)?;

    let (d_nodes, d_base_consts, d_ext_consts, d_roots, d_rap, d_alpha, d_offset) = {
        (
            stream.clone_htod(nodes)?,
            stream.clone_htod(base_consts)?,
            stream.clone_htod(ext_consts)?,
            stream.clone_htod(roots)?,
            stream.clone_htod(rap_challenges)?,
            stream.clone_htod(alpha_powers)?,
            stream.clone_htod(table_offset)?,
        )
    };

    let (d_beta_trans, d_z_inv, d_b_col, d_b_is_aux, d_b_value, d_b_beta) = {
        (
            stream.clone_htod(accum.beta_trans)?,
            stream.clone_htod(accum.z_inv)?,
            stream.clone_htod(accum.b_col)?,
            stream.clone_htod(accum.b_is_aux)?,
            stream.clone_htod(accum.b_value)?,
            stream.clone_htod(accum.b_beta)?,
        )
    };
    // D2D from the resident per-constraint columns into the flat
    // `b * num_rows + row` device layout — no PCIe, no flattened host copy,
    // no zeroing (the copies cover every element the kernel reads).
    let mut d_b_z_inv = unsafe { stream.alloc::<u64>((num_boundary * num_rows).max(1)) }?;
    {
        for (b, src) in accum.b_z_inv.iter().enumerate() {
            // Hard assert: a shorter column would leave the window's tail as
            // uninitialized VRAM the kernel reads — a silently wrong H.
            assert_eq!(src.len(), num_rows, "b_z_inv column length");
            let mut dst = d_b_z_inv.slice_mut(b * num_rows..(b + 1) * num_rows);
            stream.memcpy_dtod(&src.buf, &mut dst)?;
        }
    }

    let max_grid = MAX_THREADS / BLOCK_DIM;
    let grid = (num_rows as u32).div_ceil(BLOCK_DIM).clamp(1, max_grid);
    let num_threads = (grid as usize) * (BLOCK_DIM as usize);

    // Per-thread slot scratch, uninitialized (the walk writes before reading).
    let mut d_vals_base = unsafe { stream.alloc::<u64>((num_base_slots * num_threads).max(1)) }?;
    let mut d_vals_ext = unsafe { stream.alloc::<u64>((num_ext_slots * 3 * num_threads).max(1)) }?;
    // Output: every row is written by the grid-stride loop.
    let mut d_h = unsafe { stream.alloc::<u64>(num_rows * 3) }?;

    let num_nodes_u64 = num_nodes as u64;
    let num_roots_u64 = num_roots as u64;
    let main_stride = main.lde_size as u64;
    let aux_stride = aux.lde_size as u64;
    let next_step_u64 = next_step as u64;
    let num_rows_u64 = num_rows as u64;
    let z_len_u64 = (accum.z_inv.len() as u64).max(1);
    let num_boundary_u64 = num_boundary as u64;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.constraint_composition_kernel)
            .arg(&mut d_h)
            .arg(&d_nodes)
            .arg(&num_nodes_u64)
            .arg(&d_base_consts)
            .arg(&d_ext_consts)
            .arg(&d_roots)
            .arg(&num_roots_u64)
            .arg(&d_rap)
            .arg(&d_alpha)
            .arg(&d_offset)
            .arg(main.buf.as_ref())
            .arg(&main_stride)
            .arg(aux.buf.as_ref())
            .arg(&aux_stride)
            .arg(&next_step_u64)
            .arg(&num_rows_u64)
            .arg(&d_beta_trans)
            .arg(&d_z_inv)
            .arg(&z_len_u64)
            .arg(&num_boundary_u64)
            .arg(&d_b_col)
            .arg(&d_b_is_aux)
            .arg(&d_b_value)
            .arg(&d_b_beta)
            .arg(&d_b_z_inv)
            .arg(&mut d_vals_base)
            .arg(&mut d_vals_ext)
            .launch(cfg)?;
    }
    Ok((d_h, stream))
}

/// Evaluate the constraints AND fuse the composition accumulation on-device:
/// `H(row) = z_inv[row]·Σ βᵢ·Cᵢ + Σ_b z_b_inv[row]·β_b·(trace_b − value_b)`,
/// returning `H` as raw ext3 limbs (`num_rows * 3` u64, `out[row*3 + k]`). No
/// per-constraint matrix is materialized.
///
/// Uniform-zerofier case only (the VM has no end-exemptions); the caller gates
/// on `is_uniform` and falls back to CPU otherwise.
#[allow(clippy::too_many_arguments)]
pub fn eval_composition_on_device(
    nodes: &[u64],
    num_nodes: usize,
    num_base_slots: usize,
    num_ext_slots: usize,
    base_consts: &[u64],
    ext_consts: &[u64],
    roots: &[u64],
    rap_challenges: &[u64],
    alpha_powers: &[u64],
    table_offset: &[u64],
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    next_step: usize,
    num_rows: usize,
    accum: &CompositionAccum,
) -> Result<Vec<u64>> {
    if num_rows == 0 {
        return Ok(Vec::new());
    }
    let (d_h, stream) = eval_composition_launch(
        nodes,
        num_nodes,
        num_base_slots,
        num_ext_slots,
        base_consts,
        ext_consts,
        roots,
        rap_challenges,
        alpha_powers,
        table_offset,
        main,
        aux,
        next_step,
        num_rows,
        accum,
    )?;
    let be = backend()?;
    let pending =
        crate::device::async_dtoh_via(&stream, be.pinned_staging(), &be.ctx, &d_h, d_h.len())?;
    let mut out = vec![0u64; d_h.len()];
    pending.wait_into_u64(&mut out)?;
    Ok(out)
}

/// The composition evals `H` resident on device (interleaved ext3,
/// `num_rows * 3` u64), with the stream that produced them: downstream device
/// consumers enqueue on the same stream for ordering.
pub struct GpuCompH {
    buf: CudaSlice<u64>,
    pub num_rows: usize,
    stream: Arc<CudaStream>,
}

/// [`eval_composition_on_device`] keeping `H` on device — no D2H.
#[allow(clippy::too_many_arguments)]
pub fn eval_composition_on_device_keep(
    nodes: &[u64],
    num_nodes: usize,
    num_base_slots: usize,
    num_ext_slots: usize,
    base_consts: &[u64],
    ext_consts: &[u64],
    roots: &[u64],
    rap_challenges: &[u64],
    alpha_powers: &[u64],
    table_offset: &[u64],
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    next_step: usize,
    num_rows: usize,
    accum: &CompositionAccum,
) -> Result<GpuCompH> {
    let (buf, stream) = eval_composition_launch(
        nodes,
        num_nodes,
        num_base_slots,
        num_ext_slots,
        base_consts,
        ext_consts,
        roots,
        rap_challenges,
        alpha_powers,
        table_offset,
        main,
        aux,
        next_step,
        num_rows,
        accum,
    )?;
    Ok(GpuCompH {
        buf,
        num_rows,
        stream,
    })
}

/// D2H a resident `H` (the CPU-decompose fallback bridge).
pub fn download_comp_h(h: &GpuCompH) -> Result<Vec<u64>> {
    let be = backend()?;
    let pending = crate::device::async_dtoh_via(
        &h.stream,
        be.pinned_staging(),
        &be.ctx,
        &h.buf,
        h.buf.len(),
    )?;
    let mut out = vec![0u64; h.buf.len()];
    pending.wait_into_u64(&mut out)?;
    Ok(out)
}

/// Degree-2 quotient decomposition on device: splits a resident `H` (2n rows)
/// into the two halves `H0/H1`, written in zero-padded slab layout (6 slabs of
/// `lde_size = 2n` u64, first `n` filled) ready for the batched slab LDE.
/// Returns the slab buffer, the producing stream, and `n`.
pub fn decompose_d2_into_slabs(
    h: &GpuCompH,
    inv_2x: &GpuBaseVec,
    two_inv: u64,
) -> Result<(CudaSlice<u64>, Arc<CudaStream>, usize)> {
    let n = h.num_rows / 2;
    assert_eq!(h.num_rows, n * 2, "H row count must be even");
    assert!(inv_2x.len() >= n, "inv_2x must cover the half domain");
    let lde_size = h.num_rows;
    let be = backend()?;
    let stream = h.stream.clone();
    let mut out = stream.alloc_zeros::<u64>(6 * lde_size)?;

    let grid = (n as u32).div_ceil(BLOCK_DIM).clamp(1, MAX_THREADS / BLOCK_DIM);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u64 = n as u64;
    let stride_u64 = lde_size as u64;
    unsafe {
        stream
            .launch_builder(&be.decompose_d2_kernel)
            .arg(&h.buf)
            .arg(&inv_2x.buf)
            .arg(&two_inv)
            .arg(&n_u64)
            .arg(&stride_u64)
            .arg(&mut out)
            .launch(cfg)?;
    }
    Ok((out, stream, n))
}
