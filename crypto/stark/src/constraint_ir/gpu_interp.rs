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

pub(crate) type GoldilocksBZInv = std::sync::Arc<Vec<FieldElement<GoldilocksField>>>;

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

/// Resolve a host base-field column to its device-resident copy, uploading
/// once per distinct Arc. Returns `None` on upload failure (→ CPU fallback).
pub(crate) fn base_vec_device_handle(
    v: &GoldilocksBZInv,
) -> Option<std::sync::Arc<math_cuda::constraint_interp::GpuBaseVec>> {
    let key = std::sync::Arc::as_ptr(v) as usize;
    if let Some((_, h)) = bzinv_device_cache().lock().unwrap().get(&key) {
        return Some(h.clone());
    }
    // SAFETY: `F == GoldilocksField` by the type alias.
    let raw = unsafe { base_slice_as_u64(v.as_slice()) };
    let h = std::sync::Arc::new(math_cuda::constraint_interp::upload_base_vec(raw).ok()?);
    bzinv_device_cache()
        .lock()
        .unwrap()
        .insert(key, (v.clone(), h.clone()));
    Some(h)
}

fn bzinv_device_handles(
    vecs: &[GoldilocksBZInv],
) -> Option<Vec<std::sync::Arc<math_cuda::constraint_interp::GpuBaseVec>>> {
    vecs.iter().map(base_vec_device_handle).collect()
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
    /// The compiled composition kernel generated for this program's structure
    /// ([`super::codegen::structural_key`]), if the generator emitted one.
    compiled: Option<&'static str>,
}

/// `LAMBDA_VM_GPU_COMPILED_CONSTRAINTS`: unset, empty or `1` (the default)
/// evaluates the composition of every program that has a compiled kernel with
/// it (`crypto/math-cuda/kernels/constraint_compiled.cu`) instead of the
/// interpreter; `0` keeps the interpreter for every program. Anything else
/// stops the run.
///
/// ⛔ WHY. The interpreter reads every node from memory and every operand from,
/// and every result to, a per-thread slot file in global memory, which caps its
/// grid at 65,536 threads (0.30 waves on the 5090) and binds it on L2 (G5's
/// ncu: 77 % of L2, 30 % issue). A compiled kernel is the same node walk as
/// straight-line code over registers, with a grid that fills the card. It
/// computes the same `H` bit for bit: see [`super::codegen`].
pub const COMPILED_CONSTRAINTS_ENV: &str = "LAMBDA_VM_GPU_COMPILED_CONSTRAINTS";

/// [`COMPILED_CONSTRAINTS_ENV`] for a raw value.
pub fn compiled_constraints_setting(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("") | Some("1") => true,
        Some("0") => false,
        Some(other) => panic!("{COMPILED_CONSTRAINTS_ENV} must be 0 or 1, got {other:?}"),
    }
}

/// A test's or benchmark's override of the switch: 0 none, 1 off, 2 on.
static COMPILED_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force the compiled kernels on or off for this process (`None` returns to
/// [`COMPILED_CONSTRAINTS_ENV`]). For tests and benchmarks that compare the
/// two kernels in one process.
pub fn override_compiled_constraints(on: Option<bool>) {
    COMPILED_OVERRIDE.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::SeqCst,
    );
}

/// Whether compositions run their compiled kernel when one exists.
pub fn compiled_constraints_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    match COMPILED_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
        1 => false,
        2 => true,
        _ => *ENV.get_or_init(|| {
            let on = compiled_constraints_setting(
                std::env::var(COMPILED_CONSTRAINTS_ENV).ok().as_deref(),
            );
            let line = if on {
                format!(
                    "[gpu] constraint composition: compiled kernels for {} programs \
                     (the default; {COMPILED_CONSTRAINTS_ENV}=0 opts out), the interpreter \
                     for the rest\n",
                    math_cuda::constraint_compiled_keys::COMPILED_COMPOSITION_KERNELS.len()
                )
            } else {
                format!(
                    "[gpu] constraint composition: the interpreter \
                     ({COMPILED_CONSTRAINTS_ENV}=0)\n"
                )
            };
            use std::io::Write;
            let _ = std::io::stderr().write_all(line.as_bytes());
            on
        }),
    }
}

/// Compositions evaluated by a compiled kernel, process-wide.
pub static GPU_COMPOSITION_COMPILED_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// A test's substitute for one compiled kernel (`test-utils` builds only):
/// `(kernel, substitute)` launches `substitute` wherever `kernel` would run.
/// The proof-bytes test's mutation control swaps in a kernel with one node
/// wrong (`codegen::mutant_composition_kernel`).
#[cfg(feature = "test-utils")]
static COMPILED_SUBSTITUTE: std::sync::Mutex<Option<(&'static str, &'static str)>> =
    std::sync::Mutex::new(None);

/// Set (`Some((kernel, substitute))`) or clear the substitute kernel.
#[cfg(feature = "test-utils")]
pub fn substitute_compiled_kernel(sub: Option<(&'static str, &'static str)>) {
    *COMPILED_SUBSTITUTE
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = sub;
}

/// Compositions evaluated by a substitute kernel, process-wide.
#[cfg(feature = "test-utils")]
pub static GPU_COMPOSITION_SUBSTITUTE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
            let key = super::codegen::structural_key(&dev);
            let compiled = math_cuda::constraint_interp::compiled_composition_kernel(key);
            if compiled_constraints_enabled() {
                // One line per distinct program, so a log shows which ran compiled.
                println!(
                    "[gpu] compiled composition: {} for a {}-node program (key {key:016x})",
                    compiled.unwrap_or("none, the interpreter"),
                    dev.nodes.len()
                );
            }
            let low = std::sync::Arc::new(LoweredProgram {
                dev,
                nodes,
                ext_consts,
                roots,
                compiled,
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

/// The result of a fused GPU composition evaluation: `H` downloaded to host
/// (raw ext3 limbs, `num_rows * 3` u64) or kept resident on device for the
/// on-device degree-2 decomposition.
pub enum GpuComposition {
    Host(Vec<u64>),
    Dev(math_cuda::constraint_interp::GpuCompH),
}

/// Fused composition-poly evaluation on the GPU: returns `H(row)` (host or
/// device-resident per `keep`), or `None` for non-Goldilocks towers (→ CPU
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
    keep: bool,
) -> Option<GpuComposition>
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
    let b_z_inv_conc: &[GoldilocksBZInv] = unsafe { &*(inputs.b_z_inv as *const _ as *const _) };
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

    let compiled = lowered.compiled.filter(|_| compiled_constraints_enabled());
    #[cfg(feature = "test-utils")]
    let (compiled, substituted) = match (
        compiled,
        *COMPILED_SUBSTITUTE
            .lock()
            .unwrap_or_else(|e| e.into_inner()),
    ) {
        (Some(k), Some((from, to))) if k == from => (Some(to), true),
        (k, _) => (k, false),
    };
    let result = if keep {
        math_cuda::constraint_interp::eval_composition_on_device_keep(
            compiled,
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
        )
        .map(GpuComposition::Dev)
    } else {
        math_cuda::constraint_interp::eval_composition_on_device(
            compiled,
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
        )
        .map(GpuComposition::Host)
    };
    if result.is_ok() {
        crate::gpu_lde::GPU_COMPOSITION_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if compiled.is_some() {
            GPU_COMPOSITION_COMPILED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "test-utils")]
        if substituted {
            GPU_COMPOSITION_SUBSTITUTE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
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

#[cfg(test)]
mod tests {
    use super::compiled_constraints_setting;

    /// The compiled kernels are the default; only `0` keeps the interpreter.
    #[test]
    fn compiled_constraints_are_the_default_and_zero_opts_out() {
        assert!(compiled_constraints_setting(None));
        assert!(compiled_constraints_setting(Some("")));
        assert!(compiled_constraints_setting(Some(" 1 ")));
        assert!(!compiled_constraints_setting(Some("0")));
        assert!(!compiled_constraints_setting(Some(" 0\n")));
    }

    #[test]
    #[should_panic(expected = "must be 0 or 1")]
    fn compiled_constraints_setting_refuses_other_values() {
        compiled_constraints_setting(Some("on"));
    }
}
