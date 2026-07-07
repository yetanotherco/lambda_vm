//! GPU dispatch for the constraint interpreter (the device edge).
//!
//! Lowers a captured [`ConstraintProgram`] to its flat device blob
//! ([`DeviceProgram`]), flattens it plus the per-proof uniforms into the raw
//! `u64` slices the kernel reads, and launches
//! [`math_cuda::constraint_interp::eval_constraints_on_device`] over the
//! device-resident LDE. Returns the per-constraint eval matrix, or `None` to
//! signal the caller to fall back to the CPU path.
//!
//! This is the *one* concrete-Goldilocks lowering point: the IR is field-generic
//! everywhere else, and genericity does not cross to CUDA. A `TypeId` gate
//! establishes `F = GoldilocksField` / `E = Degree3GoldilocksExtensionField`
//! before a single `unsafe` reinterpret to the concrete program — the same
//! device-edge seam `crate::gpu_lde` uses for the LDE.
//!
//! The whole module is `#[cfg(feature = "cuda")]`; without the feature the
//! caller only ever has the CPU interpreter.

use std::any::TypeId;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;

use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};

use super::device::DeviceProgram;
use super::ir::ConstraintProgram;

/// Pack the lowered node list into 2 `u64` per node (`op | a<<32`, `b | dim<<32`),
/// the encoding the kernel's `load_node` decodes.
fn pack_nodes(dev: &DeviceProgram) -> Vec<u64> {
    let mut out = Vec::with_capacity(dev.nodes.len() * 2);
    for n in &dev.nodes {
        out.push(n.op as u64 | ((n.a as u64) << 32));
        out.push(n.b as u64 | ((n.dim as u64) << 32));
    }
    out
}

/// Flatten `[[u64; 3]]` ext3 limbs to a contiguous `u64` slice.
fn flatten_ext3(xs: &[[u64; 3]]) -> Vec<u64> {
    xs.iter().flat_map(|e| e.iter().copied()).collect()
}

/// Reinterpret a slice of ext3 field elements as flat `u64` (3 per element).
///
/// # Safety
/// The caller must have established `E == Degree3GoldilocksExtensionField`,
/// whose `FieldElement` is `#[repr(transparent)]` over `[u64; 3]` — the same
/// invariant `crate::gpu_lde::columns_to_u64_ext3` relies on.
unsafe fn ext3_slice_to_u64<E: IsField>(xs: &[FieldElement<E>]) -> Vec<u64> {
    let raw = unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u64, xs.len() * 3) };
    raw.to_vec()
}

/// Reinterpret a slice of base field elements as flat `u64` (1 per element).
///
/// # Safety
/// The caller must have established `F == GoldilocksField`, whose `FieldElement`
/// is `#[repr(transparent)]` over `u64`.
unsafe fn base_slice_to_u64<F: IsField>(xs: &[FieldElement<F>]) -> Vec<u64> {
    let raw = unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u64, xs.len()) };
    raw.to_vec()
}

/// Per-proof accumulation inputs (in `FieldElement` form) for
/// [`try_eval_composition_gpu`], mirroring the CPU accumulation in
/// `constraints::evaluator`.
pub struct CompositionInputs<'a, F: IsField, E: IsField> {
    /// Transition coefficients β, one per constraint root.
    pub beta_trans: &'a [FieldElement<E>],
    /// Cyclic transition-zerofier inverse (base field, `blowup`-length).
    pub z_inv: &'a [FieldElement<F>],
    /// Boundary constraint columns.
    pub b_col: &'a [usize],
    /// Boundary main/aux selector.
    pub b_is_aux: &'a [bool],
    /// Boundary target values.
    pub b_value: &'a [FieldElement<E>],
    /// Boundary coefficients β_b.
    pub b_beta: &'a [FieldElement<E>],
    /// Boundary zerofier inverses (base field), laid out `b * num_rows + row`.
    pub b_z_inv: &'a [FieldElement<F>],
}

/// Fused composition-poly evaluation on the GPU: returns `H(row)` as raw ext3
/// limbs (`num_rows * 3` u64), or `None` for non-Goldilocks towers (→ CPU
/// fallback). `H(row) = z_inv·Σβᵢ·Cᵢ + Σ_b z_b_inv·β_b·(trace_b − value_b)`,
/// the uniform-zerofier accumulation of `evaluator::evaluate`.
#[allow(clippy::too_many_arguments)]
pub fn try_eval_composition_gpu<F, E>(
    prog: &ConstraintProgram<F, E>,
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    rap_challenges: &[FieldElement<E>],
    alpha_powers: &[FieldElement<E>],
    table_offset: &FieldElement<E>,
    next_step: usize,
    num_rows: usize,
    inputs: &CompositionInputs<F, E>,
) -> Option<Vec<u64>>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<E>() != TypeId::of::<GoldilocksExtension>()
    {
        return None;
    }

    // SAFETY: the TypeId gate established the concrete Goldilocks tower.
    let prog: &ConstraintProgram<GoldilocksField, GoldilocksExtension> =
        unsafe { &*(prog as *const _ as *const _) };

    let dev = DeviceProgram::lower(prog);
    let nodes = pack_nodes(&dev);
    let ext_consts = flatten_ext3(&dev.ext_consts);
    let roots: Vec<u64> = dev.roots.iter().map(|&r| r as u64).collect();

    // SAFETY: `E`/`F` are the Goldilocks tower (gated above).
    let rap = unsafe { ext3_slice_to_u64(rap_challenges) };
    let alpha = unsafe { ext3_slice_to_u64(alpha_powers) };
    let offset = unsafe { ext3_slice_to_u64(std::slice::from_ref(table_offset)) };

    let beta_trans = unsafe { ext3_slice_to_u64(inputs.beta_trans) };
    let z_inv = unsafe { base_slice_to_u64(inputs.z_inv) };
    let b_value = unsafe { ext3_slice_to_u64(inputs.b_value) };
    let b_beta = unsafe { ext3_slice_to_u64(inputs.b_beta) };
    let b_z_inv = unsafe { base_slice_to_u64(inputs.b_z_inv) };
    let b_col: Vec<u64> = inputs.b_col.iter().map(|&c| c as u64).collect();
    let b_is_aux: Vec<u64> = inputs.b_is_aux.iter().map(|&a| a as u64).collect();

    let accum = math_cuda::constraint_interp::CompositionAccum {
        beta_trans: &beta_trans,
        z_inv: &z_inv,
        b_col: &b_col,
        b_is_aux: &b_is_aux,
        b_value: &b_value,
        b_beta: &b_beta,
        b_z_inv: &b_z_inv,
    };

    math_cuda::constraint_interp::eval_composition_on_device(
        &nodes,
        dev.nodes.len(),
        &dev.base_consts,
        &ext_consts,
        &roots,
        &rap,
        &alpha,
        &offset,
        main,
        aux,
        next_step,
        num_rows,
        &accum,
    )
    .ok()
}

/// Evaluate a captured program on the GPU, returning the per-constraint eval
/// matrix as raw ext3 limbs (constraint-major: constraint `c`, row `r`,
/// component `k` at `out[(c * num_rows + r) * 3 + k]`), or `None` if the field
/// tower is not the Goldilocks/degree-3 pair (→ CPU fallback).
///
/// `main`/`aux` are the device-resident LDE handles; `rap_challenges`,
/// `alpha_powers`, `table_offset` are the per-proof uniforms; `next_step` is the
/// LDE row stride for a frame-offset step; `num_rows` is the LDE row count.
#[allow(clippy::too_many_arguments)]
pub fn try_eval_program_gpu<F, E>(
    prog: &ConstraintProgram<F, E>,
    main: &GpuLdeBase,
    aux: &GpuLdeExt3,
    rap_challenges: &[FieldElement<E>],
    alpha_powers: &[FieldElement<E>],
    table_offset: &FieldElement<E>,
    next_step: usize,
    num_rows: usize,
) -> Option<Vec<u64>>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<E>() != TypeId::of::<GoldilocksExtension>()
    {
        return None;
    }

    // SAFETY: the TypeId gate above established `F = GoldilocksField` and
    // `E = Degree3GoldilocksExtensionField`, so the generic program has the
    // exact layout of the concrete one (constants are `#[repr(transparent)]`
    // over `u64` / `[u64; 3]`).
    let prog: &ConstraintProgram<GoldilocksField, GoldilocksExtension> =
        unsafe { &*(prog as *const _ as *const _) };

    let dev = DeviceProgram::lower(prog);
    let nodes = pack_nodes(&dev);
    let ext_consts = flatten_ext3(&dev.ext_consts);
    let roots: Vec<u64> = dev.roots.iter().map(|&r| r as u64).collect();

    // SAFETY: `E` is the ext3 tower (gated above).
    let rap = unsafe { ext3_slice_to_u64(rap_challenges) };
    let alpha = unsafe { ext3_slice_to_u64(alpha_powers) };
    let offset = unsafe { ext3_slice_to_u64(std::slice::from_ref(table_offset)) };

    let result = math_cuda::constraint_interp::eval_constraints_on_device(
        &nodes,
        dev.nodes.len(),
        &dev.base_consts,
        &ext_consts,
        &roots,
        &rap,
        &alpha,
        &offset,
        main,
        aux,
        next_step,
        num_rows,
    );

    // Any device error is mapped to a CPU fallback, never propagated.
    result.ok()
}
