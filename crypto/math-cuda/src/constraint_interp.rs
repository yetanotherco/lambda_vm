//! Host wrapper for the transition-constraint interpreter kernel
//! (`kernels/constraint_interp.cu`).
//!
//! Takes a constraint program already lowered to flat `u64` device arrays (by
//! `stark::constraint_ir::device::DeviceProgram`) plus the device-resident LDE
//! handles, uploads the program + per-proof uniforms, launches the interpreter
//! over every LDE row, and returns the per-constraint eval matrix.
//!
//! Layering note: this crate cannot see `stark`'s `DeviceProgram` type (stark
//! depends on math-cuda, not the reverse), so the caller flattens the program
//! into the raw `u64` slices below. The stark-side dispatch
//! (`stark::constraint_ir::gpu_interp`) owns that flattening + the TypeId gate.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::lde::{GpuLdeBase, GpuLdeExt3};

const BLOCK_DIM: u32 = 256;
/// Cap on total threads (grid × block). Each thread owns `num_nodes` ext3 slots
/// in the global value scratch (`num_nodes × MAX_THREADS × 24 B`), so a fixed
/// cap bounds that buffer regardless of LDE size; threads grid-stride over the
/// remaining rows. 65536 mirrors OpenVM's quotient `TASK_SIZE`.
const MAX_THREADS: u32 = 1 << 16;

/// Evaluate every constraint of a lowered program over the device-resident LDE.
///
/// Returns the per-constraint eval matrix as raw ext3 limbs, constraint-major:
/// constraint `c`, row `r`, component `k` at `out[(c * num_rows + r) * 3 + k]`.
/// Base-rooted constraints carry their value in component 0.
///
/// Inputs (all raw limbs, matching the crate's u64 device convention):
/// - `nodes`: 2 `u64` per IR node (`op | a<<32`, then `b | dim<<32`).
/// - `base_consts`: one `u64` per base constant.
/// - `ext_consts`, `rap_challenges`, `alpha_powers`: 3 `u64` per element.
/// - `table_offset`: exactly 3 `u64`.
/// - `roots`: one `u64` node id per constraint.
/// - `main`/`aux`: device-resident LDE handles; `next_step` is the LDE row
///   stride for a frame-offset step; `num_rows` is the number of LDE rows.
#[allow(clippy::too_many_arguments)]
pub fn eval_constraints_on_device(
    nodes: &[u64],
    num_nodes: usize,
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

    // Per-thread value scratch ([num_nodes * num_threads] ext3) and output.
    let mut d_values = stream.alloc_zeros::<u64>(num_nodes * num_threads * 3)?;
    let mut d_evals = stream.alloc_zeros::<u64>(num_roots * num_rows * 3)?;

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
            .arg(&mut d_values)
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
    /// Boundary zerofier inverses, base field: one `num_rows`-length slice per
    /// boundary constraint. Uploaded slice-by-slice into one device buffer
    /// (kernel indexing `b * num_rows + row`), so the caller never materializes
    /// a flattened host copy.
    pub b_z_inv: &'a [&'a [u64]],
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
    let num_roots = roots.len();
    if num_rows == 0 {
        return Ok(Vec::new());
    }
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
    // Per-slice upload straight from the caller's per-constraint vectors into
    // the flat `b * num_rows + row` device layout — no flattened host copy.
    let mut d_b_z_inv = stream.alloc_zeros::<u64>((num_boundary * num_rows).max(1))?;
    {
        for (b, slice) in accum.b_z_inv.iter().enumerate() {
            let mut dst = d_b_z_inv.slice_mut(b * num_rows..(b + 1) * num_rows);
            stream.memcpy_htod(*slice, &mut dst)?;
        }
    }

    let max_grid = MAX_THREADS / BLOCK_DIM;
    let grid = (num_rows as u32).div_ceil(BLOCK_DIM).clamp(1, max_grid);
    let num_threads = (grid as usize) * (BLOCK_DIM as usize);

    let mut d_values = stream.alloc_zeros::<u64>(num_nodes * num_threads * 3)?;
    let mut d_h = stream.alloc_zeros::<u64>(num_rows * 3)?;

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
            .arg(&mut d_values)
            .launch(cfg)?;
    }
    let out = {
        let pending =
            crate::device::async_dtoh_via(&stream, be.pinned_staging(), &be.ctx, &d_h, d_h.len())?;
        let mut out = vec![0u64; d_h.len()];
        pending.wait_into_u64(&mut out)?;
        out
    };
    Ok(out)
}
