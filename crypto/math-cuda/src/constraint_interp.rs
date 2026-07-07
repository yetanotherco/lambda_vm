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

    // Upload the program + uniforms (the column data never crosses PCIe — it is
    // already resident in `main.buf` / `aux.buf`).
    let d_nodes = stream.clone_htod(nodes)?;
    let d_base_consts = stream.clone_htod(base_consts)?;
    let d_ext_consts = stream.clone_htod(ext_consts)?;
    let d_roots = stream.clone_htod(roots)?;
    let d_rap = stream.clone_htod(rap_challenges)?;
    let d_alpha = stream.clone_htod(alpha_powers)?;
    let d_offset = stream.clone_htod(table_offset)?;

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
    let out = stream.clone_dtoh(&d_evals)?;
    stream.synchronize()?;
    Ok(out)
}
