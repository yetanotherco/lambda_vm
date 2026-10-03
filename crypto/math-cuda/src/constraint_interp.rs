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

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::Result;
use crate::device::backend;
use crate::lde::{GpuLdeBase, GpuLdeExt3};

const BLOCK_DIM: u32 = 256;
/// Cap on total threads (grid × block). Each thread owns `num_base_slots` u64
/// plus `num_ext_slots` ext3 slots of global value scratch, so a fixed cap
/// bounds those buffers regardless of LDE size; threads grid-stride over the
/// remaining rows. 65536 mirrors OpenVM's quotient `TASK_SIZE`.
const MAX_THREADS: u32 = 1 << 16;

/// The compiled composition kernels (`kernels/constraint_compiled.cu`), loaded
/// on first use rather than with the backend: a missing or unloadable module
/// only sends programs back to the interpreter.
const CONSTRAINT_COMPILED_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/constraint_compiled.cubin"));

/// Threads per block of the compiled kernels (`CCOMP_BLOCK` in the generated
/// source; `stark::constraint_ir::codegen::COMPILED_BLOCK_DIM`).
const COMPILED_BLOCK_DIM: u32 = 128;

/// The largest grid a compiled kernel launches; it grid-strides beyond.
const COMPILED_MAX_GRID: u32 = 65_535;

/// The compiled composition kernel generated for the program with this
/// structural key (`stark::constraint_ir::codegen::structural_key`), if any.
pub fn compiled_composition_kernel(key: u64) -> Option<&'static str> {
    let table = crate::constraint_compiled_keys::COMPILED_COMPOSITION_KERNELS;
    table
        .binary_search_by_key(&key, |&(k, _)| k)
        .ok()
        .map(|i| table[i].1)
}

/// A compiled kernel by name, from the module loaded on first use.
fn compiled_function(name: &'static str) -> Result<CudaFunction> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static MODULE: OnceLock<std::result::Result<Arc<CudaModule>, cudarc::driver::DriverError>> =
        OnceLock::new();
    static FUNCTIONS: OnceLock<Mutex<HashMap<&'static str, CudaFunction>>> = OnceLock::new();
    let module = MODULE
        .get_or_init(|| {
            let be = backend()?;
            be.ctx
                .load_module(Ptx::from_binary(CONSTRAINT_COMPILED_CUBIN.to_vec()))
        })
        .clone()?;
    let mut functions = FUNCTIONS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(f) = functions.get(name) {
        return Ok(f.clone());
    }
    let f = module.load_function(name)?;
    functions.insert(name, f.clone());
    Ok(f)
}

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
    let (d_nodes, d_base_consts, d_ext_consts, d_roots, d_rap, d_alpha, d_offset) = (
        stream.clone_htod(nodes)?,
        stream.clone_htod(base_consts)?,
        stream.clone_htod(ext_consts)?,
        stream.clone_htod(roots)?,
        stream.clone_htod(rap_challenges)?,
        stream.clone_htod(alpha_powers)?,
        stream.clone_htod(table_offset)?,
    );

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
    crate::r2split::add_upload_bytes(8 * v.len());
    crate::r2split::timed(crate::r2split::Cat::Upload, || {
        let be = backend()?;
        let stream = be.next_stream();
        let buf = stream.clone_htod(v)?;
        stream.synchronize()?;
        Ok(GpuBaseVec { buf, len: v.len() })
    })
}

/// Launch the fused composition evaluation and return the device-resident
/// result plus its stream (shared body of [`eval_composition_on_device`] and
/// [`eval_composition_on_device_keep`]).
#[allow(clippy::too_many_arguments)]
fn eval_composition_launch(
    compiled: Option<&'static str>,
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

    let ((d_nodes, d_base_consts, d_ext_consts, d_roots, d_rap, d_alpha, d_offset), uploads) =
        crate::r2split::timed(crate::r2split::Cat::Upload, || -> Result<_> {
            Ok((
                (
                    stream.clone_htod(nodes)?,
                    stream.clone_htod(base_consts)?,
                    stream.clone_htod(ext_consts)?,
                    stream.clone_htod(roots)?,
                    stream.clone_htod(rap_challenges)?,
                    stream.clone_htod(alpha_powers)?,
                    stream.clone_htod(table_offset)?,
                ),
                (
                    stream.clone_htod(accum.beta_trans)?,
                    stream.clone_htod(accum.z_inv)?,
                    stream.clone_htod(accum.b_col)?,
                    stream.clone_htod(accum.b_is_aux)?,
                    stream.clone_htod(accum.b_value)?,
                    stream.clone_htod(accum.b_beta)?,
                ),
            ))
        })?;
    crate::r2split::add_upload_bytes(
        8 * (nodes.len()
            + base_consts.len()
            + ext_consts.len()
            + roots.len()
            + rap_challenges.len()
            + alpha_powers.len()
            + table_offset.len()
            + accum.beta_trans.len()
            + accum.z_inv.len()
            + accum.b_col.len()
            + accum.b_is_aux.len()
            + accum.b_value.len()
            + accum.b_beta.len()),
    );
    let (d_beta_trans, d_z_inv, d_b_col, d_b_is_aux, d_b_value, d_b_beta) = uploads;
    // D2D from the resident per-constraint columns into the flat
    // `b * num_rows + row` device layout — no PCIe, no flattened host copy,
    // no zeroing (the copies cover every element the kernel reads).
    let d_b_z_inv = crate::r2split::timed(crate::r2split::Cat::Alloc, || -> Result<_> {
        let mut d_b_z_inv = unsafe { stream.alloc::<u64>((num_boundary * num_rows).max(1)) }?;
        for (b, src) in accum.b_z_inv.iter().enumerate() {
            // Hard assert: a shorter column would leave the window's tail as
            // uninitialized VRAM the kernel reads — a silently wrong H.
            assert_eq!(src.len(), num_rows, "b_z_inv column length");
            let mut dst = d_b_z_inv.slice_mut(b * num_rows..(b + 1) * num_rows);
            stream.memcpy_dtod(&src.buf, &mut dst)?;
        }
        Ok(d_b_z_inv)
    })?;

    // The interpreter's grid is capped by its per-thread slot file; a compiled
    // kernel keeps its values in registers, so it has no scratch and a grid
    // that fills the card. Either way every row is written by the grid-stride
    // loop, from row-local inputs only.
    let compiled_fn = compiled.map(compiled_function).transpose()?;
    let (grid, block, num_threads) = if compiled_fn.is_some() {
        let grid = (num_rows as u32)
            .div_ceil(COMPILED_BLOCK_DIM)
            .clamp(1, COMPILED_MAX_GRID);
        (grid, COMPILED_BLOCK_DIM, 0)
    } else {
        let max_grid = MAX_THREADS / BLOCK_DIM;
        let grid = (num_rows as u32).div_ceil(BLOCK_DIM).clamp(1, max_grid);
        (grid, BLOCK_DIM, (grid as usize) * (BLOCK_DIM as usize))
    };

    // Per-thread slot scratch, uninitialized (the walk writes before reading).
    // Output: every row is written by the grid-stride loop.
    let (mut d_vals_base, mut d_vals_ext, mut d_h) =
        crate::r2split::timed(crate::r2split::Cat::Alloc, || -> Result<_> {
            Ok((
                unsafe { stream.alloc::<u64>((num_base_slots * num_threads).max(1)) }?,
                unsafe { stream.alloc::<u64>((num_ext_slots * 3 * num_threads).max(1)) }?,
                unsafe { stream.alloc::<u64>(num_rows * 3) }?,
            ))
        })?;
    if COMPOSITION_TIMING.load(std::sync::atomic::Ordering::Relaxed) {
        let bytes = 8 * (num_base_slots + 3 * num_ext_slots) * num_threads;
        COMPOSITION_SCRATCH.with(|c| c.set(c.get() + bytes as u64));
    }

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
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let kernel = compiled_fn
        .as_ref()
        .unwrap_or(&be.constraint_composition_kernel);
    let timer = CompositionTimer::start(&stream)?;
    let launch_t = crate::r2split::start();
    unsafe {
        stream
            .launch_builder(kernel)
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
    crate::r2split::charge(crate::r2split::Cat::Launch, launch_t);
    timer.stop(&stream)?;
    Ok((d_h, stream))
}

/// Evaluate the constraints AND fuse the composition accumulation on-device:
/// `H(row) = z_inv[row]·Σ βᵢ·Cᵢ + Σ_b z_b_inv[row]·β_b·(trace_b − value_b)`,
/// returning `H` as raw ext3 limbs (`num_rows * 3` u64, `out[row*3 + k]`). No
/// per-constraint matrix is materialized.
///
/// Uniform-zerofier case only (the VM has no end-exemptions); the caller gates
/// on `is_uniform` and falls back to CPU otherwise.
///
/// `compiled` names the program's compiled kernel
/// ([`compiled_composition_kernel`]) to run in place of the interpreter: the
/// same `H`, bit for bit, from straight-line code with no slot file. `None`
/// runs the interpreter.
#[allow(clippy::too_many_arguments)]
pub fn eval_composition_on_device(
    compiled: Option<&'static str>,
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
        compiled,
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

/// `LAMBDA_VM_TABLE_TIMELINE`'s composition timer: when on, every
/// composition launch on a thread records a pair of CUDA events around its
/// kernel on the composition stream, and [`take_composition_device_ms`] sums
/// the kernels' device time on that thread. Off by default; no proof change.
static COMPOSITION_TIMING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

thread_local! {
    static COMPOSITION_EVENTS: std::cell::RefCell<Vec<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Bytes of slot-file scratch the slot-file interpreter allocated on this
    /// thread since the last [`take_composition_scratch_bytes`] (timer on).
    static COMPOSITION_SCRATCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The slot-file scratch bytes (the slot-file interpreter's per-thread value
/// files, outside the VRAM gate's estimate) this thread's compositions
/// allocated since the last call, while the timer is on.
pub fn take_composition_scratch_bytes() -> u64 {
    COMPOSITION_SCRATCH.with(|c| c.replace(0))
}

/// Turn the composition timer on or off, process-wide.
pub fn set_composition_timing(on: bool) {
    COMPOSITION_TIMING.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The device milliseconds of the compositions this thread launched since the
/// last call (`None` when it launched none, or the timer is off). Waits for
/// their kernels; a table's driver reads it after the table is done, when they
/// have long finished.
pub fn take_composition_device_ms() -> Option<f64> {
    let events = COMPOSITION_EVENTS.with(|e| std::mem::take(&mut *e.borrow_mut()));
    if events.is_empty() {
        return None;
    }
    let mut ms = 0.0f64;
    for (start, end) in &events {
        ms += start.elapsed_ms(end).ok()? as f64;
    }
    Some(ms)
}

/// A start event recorded on a composition stream, when the timer is on.
struct CompositionTimer(Option<cudarc::driver::CudaEvent>);

impl CompositionTimer {
    fn start(stream: &Arc<CudaStream>) -> Result<Self> {
        if !COMPOSITION_TIMING.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(Self(None));
        }
        let be = backend()?;
        let ev = be
            .ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        ev.record(stream)?;
        Ok(Self(Some(ev)))
    }

    fn stop(self, stream: &Arc<CudaStream>) -> Result<()> {
        if let Some(start) = self.0 {
            let be = backend()?;
            let end = be
                .ctx
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            end.record(stream)?;
            COMPOSITION_EVENTS.with(|e| e.borrow_mut().push((start, end)));
        }
        Ok(())
    }
}

/// The bounded-slot interpreter's kernels (`kernels/constraint_si.cu`), loaded
/// on first use: a missing or unloadable module only sends programs back to
/// the slot-file interpreter.
const CONSTRAINT_SI_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/constraint_si.cubin"));

fn si_function(name: &'static str) -> Result<CudaFunction> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static MODULE: OnceLock<std::result::Result<Arc<CudaModule>, cudarc::driver::DriverError>> =
        OnceLock::new();
    static FUNCTIONS: OnceLock<Mutex<HashMap<&'static str, CudaFunction>>> = OnceLock::new();
    let module = MODULE
        .get_or_init(|| {
            let be = backend()?;
            be.ctx
                .load_module(Ptx::from_binary(CONSTRAINT_SI_CUBIN.to_vec()))
        })
        .clone()?;
    let mut functions = FUNCTIONS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(f) = functions.get(name) {
        return Ok(f.clone());
    }
    let f = module.load_function(name)?;
    // The shared-memory variants may take more than the default 48 KiB a
    // block: allow the card's opt-in maximum once, here, so concurrent
    // launches with different sizes never race on the attribute.
    if name.starts_with("si_smem") {
        let be = backend()?;
        be.ctx.bind_to_thread()?;
        let max = be.ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )?;
        // The staged variants hold a static tile of steps; the dynamic slots
        // get what is left of the block's maximum.
        let fixed = f.get_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        )?;
        f.set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            max - fixed,
        )?;
    }
    functions.insert(name, f.clone());
    Ok(f)
}

/// Where the bounded-slot interpreter keeps a row's words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SiStore {
    /// Dynamic shared memory, `num_words · rows_per_thread · block` u64 a block.
    Shared,
    /// A per-thread local array (L1-cached local memory) of a fixed width.
    Local,
}

/// One launch shape of the bounded-slot interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SiConfig {
    pub store: SiStore,
    /// Rows a thread walks the program for at once: 1 or 2 (shared only).
    pub rows_per_thread: u32,
    /// Threads per block.
    pub block: u32,
    /// The steps staged through shared memory a tile at a time (the `_ps`
    /// kernels; one row a thread only).
    pub staged: bool,
    /// Staged, and the trace cells of the step `SI_LOOKAHEAD` ahead
    /// prefetched into L1 (the `_pp` kernels; implies `staged`).
    pub prefetch: bool,
}

/// The local-array widths `constraint_si.cu` is built at: plain, staged, and
/// staged with the prefetch.
const SI_LOCAL_WIDTHS: [(u32, [&str; 3]); 4] = [
    (32, ["si_local_w32", "si_local_w32_ps", "si_local_w32_pp"]),
    (48, ["si_local_w48", "si_local_w48_ps", "si_local_w48_pp"]),
    (64, ["si_local_w64", "si_local_w64_ps", "si_local_w64_pp"]),
    (
        128,
        ["si_local_w128", "si_local_w128_ps", "si_local_w128_pp"],
    ),
];

impl SiConfig {
    /// The kernel for a program of `num_words` words a row, if this shape has
    /// one.
    pub fn kernel(&self, num_words: u32) -> Option<&'static str> {
        let stage = match (self.staged, self.prefetch) {
            (false, false) => 0,
            (true, false) => 1,
            (_, true) => 2,
        };
        match (self.store, self.rows_per_thread, stage) {
            (SiStore::Shared, 1, _) => {
                Some(["si_smem_r1", "si_smem_r1_ps", "si_smem_r1_pp"][stage])
            }
            (SiStore::Shared, 2, 0) => Some("si_smem_r2"),
            (SiStore::Local, 1, _) => SI_LOCAL_WIDTHS
                .iter()
                .find(|(w, _)| num_words <= *w)
                .map(|(_, k)| k[stage]),
            _ => None,
        }
    }

    /// Dynamic shared memory a block takes for `num_words` words a row.
    pub fn shared_bytes(&self, num_words: u32) -> usize {
        match self.store {
            SiStore::Shared => {
                num_words.max(1) as usize * self.rows_per_thread as usize * self.block as usize * 8
            }
            SiStore::Local => 0,
        }
    }
}

/// Blocks of `cfg` a multiprocessor keeps resident for a program of
/// `num_words` words a row (the driver's occupancy query).
pub fn si_blocks_per_sm(cfg: SiConfig, num_words: u32) -> Result<u32> {
    let name = cfg.kernel(num_words).ok_or(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_INVALID_VALUE,
    ))?;
    let func = si_function(name)?;
    let smem = cfg.shared_bytes(num_words);
    let be = backend()?;
    // The occupancy query needs the context current on this thread, and
    // cudarc does not bind it for that call.
    be.ctx.bind_to_thread()?;
    func.occupancy_max_active_blocks_per_multiprocessor(cfg.block, smem, None)
}

/// The card's facts the bounded-slot interpreter's shape depends on, one line.
pub fn si_device_report() -> Result<String> {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let be = backend()?;
    be.ctx.bind_to_thread()?;
    let at = |a| be.ctx.attribute(a);
    Ok(format!(
        "SMs {} · shared memory a block (opt-in) {} B · a multiprocessor {} B · registers a \
         multiprocessor {} · L2 {} B · threads a multiprocessor {}",
        at(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        at(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
        at(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
        at(A::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?,
        at(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
        at(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
    ))
}

/// A budgeted program in device form (`stark::constraint_ir::budgeted`).
pub struct SiProgram<'a> {
    /// Four `u32` a step (`op, a, b, dst`), then one padding step the kernel's
    /// prefetch reads.
    pub steps: &'a [u32],
    pub num_steps: usize,
    pub num_words: u32,
    pub base_consts: &'a [u64],
    /// The ext uniform table, 3 `u64` an element.
    pub ext_uniforms: &'a [u64],
}

/// [`eval_composition_on_device_keep`] on the bounded-slot interpreter: the
/// same `H`, bit for bit, from a budgeted program with no slot file in global
/// memory.
pub fn eval_composition_si_keep(
    cfg: SiConfig,
    prog: &SiProgram,
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    next_step: usize,
    num_rows: usize,
    accum: &CompositionAccum,
) -> Result<GpuCompH> {
    assert!(num_rows > 0, "callers gate empty domains");
    assert_eq!(
        prog.steps.len(),
        4 * (prog.num_steps + 1),
        "4 u32 a step, plus the padding step"
    );
    let num_boundary = accum.b_col.len();
    assert_eq!(accum.b_z_inv.len(), num_boundary, "z_b_inv per boundary");
    assert_eq!(accum.b_is_aux.len(), num_boundary, "b_is_aux per boundary");
    assert_eq!(
        accum.b_value.len(),
        num_boundary * 3,
        "b_value ext3 per boundary"
    );
    assert_eq!(
        accum.b_beta.len(),
        num_boundary * 3,
        "b_beta ext3 per boundary"
    );
    let name = cfg
        .kernel(prog.num_words)
        .ok_or(cudarc::driver::DriverError(
            cudarc::driver::sys::cudaError_enum::CUDA_ERROR_INVALID_VALUE,
        ))?;
    let func = si_function(name)?;
    let smem = cfg.shared_bytes(prog.num_words);

    let be = backend()?;
    let stream = be.next_stream();
    main.wait_ready_on(&stream)?;
    aux.wait_ready_on(&stream)?;
    let base_consts = if prog.base_consts.is_empty() {
        &[0u64][..]
    } else {
        prog.base_consts
    };
    let (
        d_steps,
        d_base_consts,
        d_uni,
        (d_beta, d_z_inv, d_b_col, d_b_is_aux, d_b_value, d_b_beta),
    ) = crate::r2split::timed(crate::r2split::Cat::Upload, || -> Result<_> {
        Ok((
            stream.clone_htod(prog.steps)?,
            stream.clone_htod(base_consts)?,
            stream.clone_htod(prog.ext_uniforms)?,
            (
                stream.clone_htod(accum.beta_trans)?,
                stream.clone_htod(accum.z_inv)?,
                stream.clone_htod(accum.b_col)?,
                stream.clone_htod(accum.b_is_aux)?,
                stream.clone_htod(accum.b_value)?,
                stream.clone_htod(accum.b_beta)?,
            ),
        ))
    })?;
    crate::r2split::add_upload_bytes(
        4 * prog.steps.len()
            + 8 * (base_consts.len()
                + prog.ext_uniforms.len()
                + accum.beta_trans.len()
                + accum.z_inv.len()
                + accum.b_col.len()
                + accum.b_is_aux.len()
                + accum.b_value.len()
                + accum.b_beta.len()),
    );
    let (d_b_z_inv, mut d_h) =
        crate::r2split::timed(crate::r2split::Cat::Alloc, || -> Result<_> {
            let mut d_b_z_inv = unsafe { stream.alloc::<u64>((num_boundary * num_rows).max(1)) }?;
            for (b, src) in accum.b_z_inv.iter().enumerate() {
                assert_eq!(src.len(), num_rows, "b_z_inv column length");
                let mut dst = d_b_z_inv.slice_mut(b * num_rows..(b + 1) * num_rows);
                stream.memcpy_dtod(&src.buf, &mut dst)?;
            }
            Ok((d_b_z_inv, unsafe { stream.alloc::<u64>(num_rows * 3) }?))
        })?;

    // One row (or two) a thread, every row covered: no slot file caps the grid.
    let rows_per_block = (cfg.block * cfg.rows_per_thread) as usize;
    let grid = num_rows
        .div_ceil(rows_per_block)
        .clamp(1, i32::MAX as usize) as u32;
    let num_steps_u32 = prog.num_steps as u32;
    let main_stride = main.lde_size as u64;
    let aux_stride = aux.lde_size as u64;
    let next_step_u64 = next_step as u64;
    let num_rows_u64 = num_rows as u64;
    let z_len_u64 = (accum.z_inv.len() as u64).max(1);
    let num_boundary_u64 = num_boundary as u64;
    let launch = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (cfg.block, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let timer = CompositionTimer::start(&stream)?;
    let launch_t = crate::r2split::start();
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut d_h)
            .arg(&d_steps)
            .arg(&num_steps_u32)
            .arg(&d_base_consts)
            .arg(&d_uni)
            .arg(&d_beta)
            .arg(main.buf.as_ref())
            .arg(&main_stride)
            .arg(aux.buf.as_ref())
            .arg(&aux_stride)
            .arg(&next_step_u64)
            .arg(&num_rows_u64)
            .arg(&d_z_inv)
            .arg(&z_len_u64)
            .arg(&num_boundary_u64)
            .arg(&d_b_col)
            .arg(&d_b_is_aux)
            .arg(&d_b_value)
            .arg(&d_b_beta)
            .arg(&d_b_z_inv)
            .launch(launch)?;
    }
    crate::r2split::charge(crate::r2split::Cat::Launch, launch_t);
    timer.stop(&stream)?;
    Ok(GpuCompH {
        buf: d_h,
        num_rows,
        stream,
    })
}

/// The composition evals `H` resident on device (interleaved ext3,
/// `num_rows * 3` u64), with the stream that produced them: downstream device
/// consumers enqueue on the same stream for ordering.
pub struct GpuCompH {
    buf: CudaSlice<u64>,
    pub num_rows: usize,
    stream: Arc<CudaStream>,
}

impl GpuCompH {
    /// Wait until `H` is computed (a benchmark's clock stop; the prover never
    /// waits here, it enqueues its consumers on the same stream).
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
}

/// [`eval_composition_on_device`] keeping `H` on device — no D2H.
#[allow(clippy::too_many_arguments)]
pub fn eval_composition_on_device_keep(
    compiled: Option<&'static str>,
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
        compiled,
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

/// H2D a host `H` (interleaved ext3, `3 * num_rows` u64) as a resident
/// [`GpuCompH`]: the input side of the decomposition parity tests, which feed
/// the device and host splits the same evaluations.
pub fn upload_comp_h(h: &[u64]) -> Result<GpuCompH> {
    assert_eq!(h.len() % 3, 0, "H is interleaved ext3");
    let be = backend()?;
    let stream = be.next_stream();
    let buf = stream.clone_htod(h)?;
    Ok(GpuCompH {
        buf,
        num_rows: h.len() / 3,
        stream,
    })
}

/// D2H a resident `H` (the CPU-decompose fallback bridge).
pub fn download_comp_h(h: &GpuCompH) -> Result<Vec<u64>> {
    use crate::r2split::{Cat, timed};
    let be = backend()?;
    let pending = timed(Cat::Stage, || {
        crate::device::async_dtoh_via(&h.stream, be.pinned_staging(), &be.ctx, &h.buf, h.buf.len())
    })?;
    let mut out = timed(Cat::Drain, || vec![0u64; h.buf.len()]);
    timed(Cat::DevWait, || pending.wait_into_u64_timed(&mut out))?;
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
    let mut out = crate::r2split::timed(crate::r2split::Cat::Alloc, || {
        stream.alloc_zeros::<u64>(6 * lde_size)
    })?;

    let grid = (n as u32)
        .div_ceil(BLOCK_DIM)
        .clamp(1, MAX_THREADS / BLOCK_DIM);
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

/// Four-part quotient decomposition on device (`decompose_d4_ext3`): splits a
/// resident `H` (`4q` rows) into `H0..H3` with `H(x) = Σ_j x^j·H_j(x^4)`,
/// written in zero-padded slab layout (12 slabs of `lde_size = 4q` u64, first
/// `q` filled, part `j` at slabs `3j..3j+3`) ready for the batched slab LDE at
/// ratio 4. `inv_2x` covers the LDE/2 coset (`1/(2·g·w^i)`), `inv_2y` the
/// LDE/4 one (`1/(2·g²·w^{2i})`). Returns the slab buffer, the producing
/// stream, and `q`.
pub fn decompose_d4_into_slabs(
    h: &GpuCompH,
    inv_2x: &GpuBaseVec,
    inv_2y: &GpuBaseVec,
    two_inv: u64,
) -> Result<(CudaSlice<u64>, Arc<CudaStream>, usize)> {
    let q = h.num_rows / 4;
    assert_eq!(h.num_rows, q * 4, "H row count must be a multiple of 4");
    assert!(inv_2x.len() >= 2 * q, "inv_2x must cover the half domain");
    assert!(inv_2y.len() >= q, "inv_2y must cover the quarter domain");
    let lde_size = h.num_rows;
    let be = backend()?;
    let stream = h.stream.clone();
    let mut out = crate::r2split::timed(crate::r2split::Cat::Alloc, || {
        stream.alloc_zeros::<u64>(12 * lde_size)
    })?;

    let grid = (q as u32)
        .div_ceil(BLOCK_DIM)
        .clamp(1, MAX_THREADS / BLOCK_DIM);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    let q_u64 = q as u64;
    let stride_u64 = lde_size as u64;
    unsafe {
        stream
            .launch_builder(&be.decompose_d4_kernel)
            .arg(&h.buf)
            .arg(&inv_2x.buf)
            .arg(&inv_2y.buf)
            .arg(&two_inv)
            .arg(&q_u64)
            .arg(&stride_u64)
            .arg(&mut out)
            .launch(cfg)?;
    }
    Ok((out, stream, q))
}

/// Degree-1 (num_parts==1) composition part: `H` is already the single part on
/// the LDE coset, so there is neither a decompose nor a re-extension — only a
/// de-interleave of the resident interleaved ext3 evals `h` (`num_rows` rows,
/// `h[row*3 + k]`) into the 3-slab layout the commit / DEEP / FRI consumers read
/// (`out[(0*3 + k) * lde_size + row]`, i.e. one column of 3 slabs). Returns a
/// device-resident [`GpuLdeExt3`] with `m = 1` and `lde_size == h.num_rows`,
/// kept live on `h`'s stream with a recorded event so cross-stream consumers
/// wait device-side (no host block).
pub fn comp_h_to_slabs(h: &GpuCompH) -> Result<GpuLdeExt3> {
    let lde_size = h.num_rows;
    assert!(
        lde_size.is_power_of_two() && lde_size >= 2,
        "H row count must be a power of two"
    );
    let be = backend()?;
    let stream = h.stream.clone();
    // The kernel writes every one of the `3 * lde_size` slab u64s, so an
    // uninitialized allocation is sound (no zero-pad tail, unlike the d=2
    // decompose which only fills the first `n` rows).
    let mut out = crate::r2split::timed(crate::r2split::Cat::Alloc, || unsafe {
        stream.alloc::<u64>(3 * lde_size)
    })?;

    let grid = (lde_size as u32)
        .div_ceil(BLOCK_DIM)
        .clamp(1, MAX_THREADS / BLOCK_DIM);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_rows_u64 = lde_size as u64;
    unsafe {
        stream
            .launch_builder(&be.comp_h_to_slabs_kernel)
            .arg(&h.buf)
            .arg(&num_rows_u64)
            .arg(&mut out)
            .launch(cfg)?;
    }

    let ready = be.take_event()?;
    ready.event().record(&stream)?;

    Ok(GpuLdeExt3 {
        buf: Arc::new(out),
        m: 1,
        lde_size,
        tree: None,
        ready: Some(Arc::new(ready)),
    })
}

/// Parity helper: build a resident [`GpuCompH`] from interleaved ext3 evals on
/// host (`h[row*3 + k]`, `num_rows * 3` u64), uploaded on a fresh stream. On the
/// prove path `H` is born on device (never uploaded); this exists only so the
/// de-interleave kernel can be exercised in isolation against a host oracle.
pub fn comp_h_from_host_interleaved(interleaved: &[u64], num_rows: usize) -> Result<GpuCompH> {
    assert_eq!(interleaved.len(), num_rows * 3, "interleaved ext3 length");
    let be = backend()?;
    let stream = be.next_stream();
    let buf = stream.clone_htod(interleaved)?;
    stream.synchronize()?;
    Ok(GpuCompH {
        buf,
        num_rows,
        stream,
    })
}
