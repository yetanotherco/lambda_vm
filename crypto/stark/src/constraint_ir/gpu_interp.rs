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

/// Pack the lowered node list into 2 `u64` per node (`op | a<<32`, `b | res<<32`),
/// the encoding the kernel's `load_node` decodes.
fn pack_nodes(dev: &DeviceProgram) -> Vec<u64> {
    let mut out = Vec::with_capacity(dev.nodes.len() * 2);
    for n in &dev.nodes {
        out.push(n.op as u64 | ((n.a as u64) << 32));
        out.push(n.b as u64 | ((n.res as u64) << 32));
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

/// Borrowing sibling of [`base_slice_to_u64`]: the same reinterpret with no
/// copy, for buffers that go straight to a device upload.
///
/// # Safety
/// Same contract: the caller must have established `F == GoldilocksField`.
unsafe fn base_slice_as_u64<F: IsField>(xs: &[FieldElement<F>]) -> &[u64] {
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u64, xs.len()) }
}

/// Lift raw base-field limbs (one canonical-Goldilocks `u64` per element, as
/// produced by the device row-gather kernels) back into owned `FieldElement`s.
/// Returns `None` unless `F == GoldilocksField`. The inverse of
/// [`base_slice_to_u64`]; used to feed device-gathered LDE rows into the generic
/// prover openings.
pub fn base_u64_to_field<F: IsField + 'static>(raw: &[u64]) -> Option<Vec<FieldElement<F>>> {
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    // SAFETY: the gate established `F == GoldilocksField`, whose `FieldElement`
    // is `#[repr(transparent)]` over `u64`; `raw` (a `*const u64`) is 8-aligned.
    let fe =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const FieldElement<F>, raw.len()) };
    Some(fe.to_vec())
}

/// Ext3 sibling of [`base_u64_to_field`]: `raw` holds `3` interleaved limbs per
/// element (`[c0, c1, c2]`). Returns `None` unless `E` is the degree-3
/// Goldilocks extension. Inverse of [`ext3_slice_to_u64`].
pub fn ext3_u64_to_field<E: IsField + 'static>(raw: &[u64]) -> Option<Vec<FieldElement<E>>> {
    if TypeId::of::<E>() != TypeId::of::<GoldilocksExtension>() {
        return None;
    }
    debug_assert_eq!(raw.len() % 3, 0, "ext3 limbs must come in triples");
    // SAFETY: the gate established the degree-3 Goldilocks extension, whose
    // `FieldElement` is `#[repr(transparent)]` over `[u64; 3]`; `raw` (a
    // `*const u64`) is 8-aligned, matching `[u64; 3]`'s alignment.
    let fe = unsafe {
        std::slice::from_raw_parts(raw.as_ptr() as *const FieldElement<E>, raw.len() / 3)
    };
    Some(fe.to_vec())
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
    /// Boundary zerofier inverses (base field): one `num_rows`-length vector
    /// per boundary constraint (constraints sharing a step share the Arc,
    /// cached per domain) — resolved to device-resident columns via
    /// [`bzinv_device_handles`], so nothing LDE-sized crosses PCIe per dispatch.
    pub b_z_inv: &'a [std::sync::Arc<Vec<FieldElement<F>>>],
}

type GoldilocksBZInv = std::sync::Arc<Vec<FieldElement<GoldilocksField>>>;

/// Device-resident boundary-zerofier columns, keyed by the host Arc
/// allocation. The entry stores the Arc, pinning the allocation: a key can
/// never be reused while its entry lives (entries live for the process, like
/// the per-domain host cache that feeds them).
#[allow(clippy::type_complexity)]
fn bzinv_device_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<
        usize,
        (
            GoldilocksBZInv,
            std::sync::Arc<math_cuda::constraint_interp::GpuBaseVec>,
        ),
    >,
> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<
                usize,
                (
                    GoldilocksBZInv,
                    std::sync::Arc<math_cuda::constraint_interp::GpuBaseVec>,
                ),
            >,
        >,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// Resolve each host column to its device-resident copy, uploading once per
/// distinct Arc. Returns `None` on any upload failure (→ CPU fallback).
fn bzinv_device_handles(
    vecs: &[GoldilocksBZInv],
) -> Option<Vec<std::sync::Arc<math_cuda::constraint_interp::GpuBaseVec>>> {
    vecs.iter()
        .map(|v| {
            let key = std::sync::Arc::as_ptr(v) as usize;
            if let Some((_, h)) = bzinv_device_cache().lock().unwrap().get(&key) {
                return Some(h.clone());
            }
            // SAFETY: `F == GoldilocksField` by the type alias.
            let raw = unsafe { base_slice_as_u64(v.as_slice()) };
            let h = std::sync::Arc::new(
                math_cuda::constraint_interp::upload_base_vec(raw).ok()?,
            );
            bzinv_device_cache()
                .lock()
                .unwrap()
                .insert(key, (v.clone(), h.clone()));
            Some(h)
        })
        .collect()
}

/// The program-derived half of a lowered call: the flat device blob plus its
/// packed program uniforms. Depends only on the program content — identical
/// across continuation epochs and table shards — so it is cached process-wide
/// (see [`lowering_cache`]).
struct LoweredProgram {
    dev: DeviceProgram,
    nodes: Vec<u64>,
    ext_consts: Vec<u64>,
    roots: Vec<u64>,
}

/// The lowered device program plus the packed per-proof uniforms shared by both
/// GPU dispatch entry points. Produced by [`lower_and_pack`].
struct LoweredCall {
    lowered: std::sync::Arc<LoweredProgram>,
    rap: Vec<u64>,
    alpha: Vec<u64>,
    offset: Vec<u64>,
}

type GoldilocksProgram = ConstraintProgram<GoldilocksField, GoldilocksExtension>;

/// Process-wide cache of lowered programs, keyed by content fingerprint. A hit
/// must pass the full-equality check against the stored snapshot — a
/// fingerprint collision re-lowers, never aliases another program.
#[allow(clippy::type_complexity)]
fn lowering_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<u64, (GoldilocksProgram, std::sync::Arc<LoweredProgram>)>,
> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<u64, (GoldilocksProgram, std::sync::Arc<LoweredProgram>)>,
        >,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn program_fingerprint(p: &GoldilocksProgram) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    p.nodes.hash(&mut h);
    p.dims.hash(&mut h);
    // The const tables lack `Hash`: hash their canonical limbs.
    // SAFETY: `p` is the concrete Goldilocks program.
    unsafe { base_slice_as_u64(&p.base_consts) }.hash(&mut h);
    unsafe { ext3_slice_to_u64(&p.ext_consts) }.hash(&mut h);
    p.roots.hash(&mut h);
    p.num_base.hash(&mut h);
    h.finish()
}

fn program_eq(a: &GoldilocksProgram, b: &GoldilocksProgram) -> bool {
    a.num_base == b.num_base
        && a.roots == b.roots
        && a.nodes == b.nodes
        && a.dims == b.dims
        && a.base_consts == b.base_consts
        && a.ext_consts == b.ext_consts
}

/// The single concrete-Goldilocks lowering seam shared by
/// [`try_eval_composition_gpu`] and [`try_eval_program_gpu`]: gate on the
/// Goldilocks tower, reinterpret the generic program once, lower it to the flat
/// device blob, and pack the three ext3 uniforms. Returns `None` (→ CPU
/// fallback) for any other field tower. Factoring this keeps the sole `unsafe`
/// program reinterpret and the TypeId gate in one place instead of two.
fn lower_and_pack<F, E>(
    prog: &ConstraintProgram<F, E>,
    rap_challenges: &[FieldElement<E>],
    alpha_powers: &[FieldElement<E>],
    table_offset: &FieldElement<E>,
) -> Option<LoweredCall>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if !crate::gpu_lde::is_goldilocks_ext3_tower::<F, E>() {
        return None;
    }
    // SAFETY: the TypeId gate established `F = GoldilocksField` and
    // `E = Degree3GoldilocksExtensionField`; the generic program has the exact
    // layout of the concrete one (constants are `#[repr(transparent)]` over
    // `u64` / `[u64; 3]`).
    let prog: &GoldilocksProgram = unsafe { &*(prog as *const _ as *const _) };

    let key = program_fingerprint(prog);
    let hit = {
        let cache = lowering_cache().lock().unwrap();
        match cache.get(&key) {
            Some((snapshot, low)) if program_eq(snapshot, prog) => Some(low.clone()),
            _ => None,
        }
    };
    let lowered = match hit {
        Some(low) => low,
        None => {
            let dev = DeviceProgram::lower(prog);
            let nodes = pack_nodes(&dev);
            let ext_consts = flatten_ext3(&dev.ext_consts);
            let roots: Vec<u64> = dev.roots.iter().map(|&r| r as u64).collect();
            let low = std::sync::Arc::new(LoweredProgram {
                dev,
                nodes,
                ext_consts,
                roots,
            });
            lowering_cache()
                .lock()
                .unwrap()
                .insert(key, (prog.clone(), low.clone()));
            low
        }
    };

    // SAFETY: `E` is the ext3 tower (gated above).
    let rap = unsafe { ext3_slice_to_u64(rap_challenges) };
    let alpha = unsafe { ext3_slice_to_u64(alpha_powers) };
    let offset = unsafe { ext3_slice_to_u64(std::slice::from_ref(table_offset)) };

    Some(LoweredCall {
        lowered,
        rap,
        alpha,
        offset,
    })
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
    let LoweredCall {
        lowered,
        rap,
        alpha,
        offset,
    } = lower_and_pack(prog, rap_challenges, alpha_powers, table_offset)?;

    // SAFETY: `E`/`F` are the Goldilocks tower (established in `lower_and_pack`).
    let beta_trans = unsafe { ext3_slice_to_u64(inputs.beta_trans) };
    let z_inv = unsafe { base_slice_to_u64(inputs.z_inv) };
    let b_value = unsafe { ext3_slice_to_u64(inputs.b_value) };
    let b_beta = unsafe { ext3_slice_to_u64(inputs.b_beta) };
    // SAFETY: `F` is Goldilocks (established in `lower_and_pack`);
    // `Vec<FieldElement<F>>` and the concrete Vec share their layout.
    let b_z_inv_conc: &[GoldilocksBZInv] =
        unsafe { &*(inputs.b_z_inv as *const _ as *const _) };
    let b_z_inv_handles = bzinv_device_handles(b_z_inv_conc)?;
    let b_z_inv: Vec<&math_cuda::constraint_interp::GpuBaseVec> =
        b_z_inv_handles.iter().map(|h| h.as_ref()).collect();
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

    let result = math_cuda::constraint_interp::eval_composition_on_device(
        &lowered.nodes,
        lowered.dev.nodes.len(),
        lowered.dev.num_base_slots as usize,
        lowered.dev.num_ext_slots as usize,
        &lowered.dev.base_consts,
        &lowered.ext_consts,
        &lowered.roots,
        &rap,
        &alpha,
        &offset,
        main,
        aux,
        next_step,
        num_rows,
        &accum,
    );
    if result.is_ok() {
        crate::gpu_lde::GPU_COMPOSITION_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    result.ok()
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
    let LoweredCall {
        lowered,
        rap,
        alpha,
        offset,
    } = lower_and_pack(prog, rap_challenges, alpha_powers, table_offset)?;

    let result = math_cuda::constraint_interp::eval_constraints_on_device(
        &lowered.nodes,
        lowered.dev.nodes.len(),
        lowered.dev.num_base_slots as usize,
        lowered.dev.num_ext_slots as usize,
        &lowered.dev.base_consts,
        &lowered.ext_consts,
        &lowered.roots,
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
