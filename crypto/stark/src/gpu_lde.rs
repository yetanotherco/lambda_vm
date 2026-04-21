//! GPU dispatch layer for the per-column coset LDE. Lives in the stark crate
//! (not `math`) to avoid a dependency cycle between `math` and `math-cuda`.
//!
//! Handles only Goldilocks base-field columns above a size threshold; falls
//! back to CPU for extension-field columns and small columns where kernel
//! launch overhead dominates. Produces the same natural-order, non-canonical
//! LDE evaluations as the CPU path.

use core::any::type_name;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsSubFieldOf};

use crate::domain::Domain;

/// Break-even LDE size. Below this, the CPU `coset_lde_full_expand` completes
/// in a few hundred microseconds and the GPU's ~37 kernel launches plus
/// H2D/D2H round-trip is a net loss. The check is on **lde size**, not trace
/// length, because that's what determines the FFT workload.
///
/// 2^19 is a conservative default calibrated against a 46-core machine where
/// rayon-parallel CPU LDE is already fast. Override via env var for tuning
/// on smaller machines; see `/workspace/lambda_vm/crypto/math-cuda/tests/bench_quick.rs`.
const DEFAULT_GPU_LDE_THRESHOLD: usize = 1 << 19;

fn gpu_lde_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_LDE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_LDE_THRESHOLD)
    })
}

/// Atomically counted by `try_expand_column` every time it actually routes a
/// column to the GPU. Used by benchmarks to confirm the GPU path fired.
static GPU_LDE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn gpu_lde_calls() -> u64 {
    GPU_LDE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn reset_gpu_lde_calls() {
    GPU_LDE_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) static GPU_EXTEND_HALVES_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub fn gpu_extend_halves_calls() -> u64 {
    GPU_EXTEND_HALVES_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Try to GPU-batch all columns in one pass.
///
/// Only engaged for Goldilocks-base tables whose LDE size is above the
/// threshold. The prover's `expand_columns_to_lde` hands us every column of
/// one table at once; those columns all share twiddles and coset weights so
/// they can be processed in a single batched pipeline on one stream.
///
/// Returns `true` if the batch was handled on GPU (and `columns` now contains
/// the LDE evaluations). Returns `false` to let the caller run the per-column
/// CPU fallback.
#[inline]
pub(crate) fn try_expand_columns_batched<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> bool
where
    F: IsField,
    E: IsField,
{
    if columns.is_empty() {
        return true; // nothing to do — same as CPU path
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return false;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return false;
    }
    // All columns within one call must be the same size (invariant of the
    // caller), but double-check before unsafe extraction.
    if columns.iter().any(|c| c.len() != n) {
        return false;
    }

    // Ext3 fast path: decompose each ext3 column into its 3 base components
    // and dispatch to the base-field batched NTT with 3×M logical columns.
    // Butterflies with a base-field twiddle act componentwise on ext3, so
    // this is exactly equivalent to running the NTT in the extension field.
    if type_name::<E>() == type_name::<Degree3GoldilocksExtensionField>() {
        return try_expand_columns_batched_ext3::<F, E>(columns, blowup_factor, weights);
    }

    if type_name::<E>() != type_name::<GoldilocksField>() {
        return false;
    }

    // Extract raw u64 slices. SAFETY: type_name above confirms
    // `E == GoldilocksField`, so `FieldElement<E>` wraps u64 one-to-one.
    let raw_columns: Vec<Vec<u64>> = columns
        .iter()
        .map(|col| {
            col.iter()
                .map(|e| unsafe { *(e.value() as *const _ as *const u64) })
                .collect()
        })
        .collect();
    let weights_u64: Vec<u64> = weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect();

    // Pre-size caller Vecs to lde_size so the GPU path can write directly
    // into the same backing allocation the caller already holds. This skips
    // the intermediate `Vec<Vec<u64>>` allocation (which would page-fault
    // per column) and is the main reason `coset_lde_batch_base_into` exists.
    for col in columns.iter_mut() {
        // SAFETY: set_len is valid here because capacity is already >=
        // lde_size (the caller sized columns via `extract_columns_main(lde_size)`)
        // and we're about to overwrite every slot via the GPU copy below.
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }

    // Borrow each caller Vec as a raw `&mut [u64]` slice; safe because each
    // FieldElement<E> aliases a single u64 when E == GoldilocksField.
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len();
            // SAFETY: see above — single-u64 layout, caller still owns.
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();
    GPU_LDE_CALLS.fetch_add(columns.len() as u64, std::sync::atomic::Ordering::Relaxed);
    math_cuda::lde::coset_lde_batch_base_into(
        &slices,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
    )
    .expect("GPU batched coset LDE failed");
    true
}

/// GPU path for `Prover::extend_half_to_lde`.
///
/// Inside `decompose_and_extend_d2` (R2 quotient decomposition) the prover
/// does `rayon::join` of two calls: `iFFT(N on g²-coset) → FFT(2N on g-coset)`
/// over ext3 halves H0 and H1. They share the same domain/offset and sizes,
/// so we batch them into a single GPU call with M=2 ext3 columns.
///
/// Weights = `[1/N, g^(-1)/N, g^(-2)/N, …, g^(-(N-1))/N]`. This bakes the
/// `(g²)^(-k)` input-coset-undo from `interpolate_offset_fft` together with
/// the `g^k` forward-coset-shift from `evaluate_polynomial_on_lde_domain` —
/// net is `g^(-k)` — plus the `1/N` iFFT normalisation.
///
/// Returns `None` when the GPU path doesn't apply (too small, or CPU path
/// should be used); in that case the caller runs its existing rayon::join.
pub(crate) fn try_extend_two_halves_gpu<F, E>(
    h0: &[FieldElement<E>],
    h1: &[FieldElement<E>],
    squared_offset: &FieldElement<F>,
    domain: &Domain<F>,
) -> Option<(Vec<FieldElement<E>>, Vec<FieldElement<E>>)>
where
    F: math::field::traits::IsFFTField + IsField,
    E: IsField,
    F: IsSubFieldOf<E>,
{
    if h0.len() != h1.len() {
        return None;
    }
    let n = h0.len();
    let blowup = 2; // extend_half_to_lde extends N → 2N always
    let lde_size = n * blowup;
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    GPU_EXTEND_HALVES_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // squared_offset should be `g²`. We recover `g` as `domain.coset_offset`
    // and use it to build the `g^(-k) / N` weights.
    let _ = squared_offset; // unused (we derive weights from domain)

    // Flatten ext3 slices to raw 3*n u64 buffers.
    let to_u64 = |col: &[FieldElement<E>]| -> Vec<u64> {
        let len = col.len() * 3;
        let ptr = col.as_ptr() as *const u64;
        unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
    };
    let h0_raw = to_u64(h0);
    let h1_raw = to_u64(h1);

    // weights[k] = g^(-k) / N as a u64.
    let inv_n = FieldElement::<F>::from(n as u64)
        .inv()
        .expect("N nonzero");
    let g = &domain.coset_offset;
    let g_inv = g.inv().expect("g nonzero");
    let mut weights_u64 = Vec::with_capacity(n);
    let mut w = inv_n.clone();
    for _ in 0..n {
        // F == GoldilocksField by type_name check above, so value is u64.
        let v: u64 = unsafe { *(w.value() as *const _ as *const u64) };
        weights_u64.push(v);
        w = w * &g_inv;
    }

    // Pre-allocate outputs.
    let mut lde_h0 = vec![FieldElement::<E>::zero(); lde_size];
    let mut lde_h1 = vec![FieldElement::<E>::zero(); lde_size];

    GPU_LDE_CALLS.fetch_add(6, std::sync::atomic::Ordering::Relaxed); // 2 ext3 cols × 3 components
    {
        let inputs: [&[u64]; 2] = [&h0_raw, &h1_raw];
        // View each output Vec<FieldElement<E>> as &mut [u64] of length 3*lde_size.
        let out0_ptr = lde_h0.as_mut_ptr() as *mut u64;
        let out1_ptr = lde_h1.as_mut_ptr() as *mut u64;
        // SAFETY: ext3 FieldElement is [u64; 3] in memory, and the Vec has len
        // = lde_size so the backing is 3*lde_size u64s.
        let out0_slice = unsafe { core::slice::from_raw_parts_mut(out0_ptr, 3 * lde_size) };
        let out1_slice = unsafe { core::slice::from_raw_parts_mut(out1_ptr, 3 * lde_size) };
        let mut outputs: [&mut [u64]; 2] = [out0_slice, out1_slice];
        math_cuda::lde::coset_lde_batch_ext3_into(
            &inputs,
            n,
            blowup,
            &weights_u64,
            &mut outputs,
        )
        .expect("GPU extend_half_to_lde failed");
    }

    Some((lde_h0, lde_h1))
}

/// GPU path for Round 4's DEEP-poly LDE extension.
///
/// The CPU pipeline at `prover.rs:1107` is
/// ```ignore
/// let deep_poly = Polynomial::interpolate_fft::<Field>(&deep_evals)?;
/// let mut lde_evals = Polynomial::evaluate_fft::<Field>(&deep_poly, 1, Some(domain_size))?;
/// in_place_bit_reverse_permute(&mut lde_evals);
/// ```
///
/// That is an iFFT over `N = deep_evals.len()` ext3 elements followed by an
/// FFT evaluation on `domain_size` points — the **standard** (non-coset) LDE
/// on the extension field with weights `[1/N, ..., 1/N]`. We reuse
/// `coset_lde_batch_ext3_into` with a uniform `1/N` weight vector; the
/// single ext3 column is handled internally as 3 base-field slabs. The
/// caller keeps its trailing `in_place_bit_reverse_permute`, so output
/// order is unchanged.
pub(crate) fn try_r4_deep_poly_lde_gpu<E>(
    deep_evals: &[FieldElement<E>],
    domain_size: usize,
) -> Option<Vec<FieldElement<E>>>
where
    E: IsField,
{
    let n = deep_evals.len();
    if n == 0 || !n.is_power_of_two() {
        return None;
    }
    if domain_size < n || !domain_size.is_power_of_two() {
        return None;
    }
    let blowup = domain_size / n;
    if blowup < 2 {
        return None;
    }
    if domain_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }

    GPU_R4_LDE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Uniform weights = 1/N (no coset shift, just iFFT normalisation).
    let inv_n_u64 = {
        let fe = FieldElement::<GoldilocksField>::from(n as u64)
            .inv()
            .expect("N non-zero");
        *fe.value()
    };
    let weights = vec![inv_n_u64; n];

    // Input: single ext3 column, 3n u64s.
    let input_raw: Vec<u64> = {
        let len = n * 3;
        let ptr = deep_evals.as_ptr() as *const u64;
        unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
    };
    let inputs: [&[u64]; 1] = [&input_raw];

    let mut out_vec = vec![FieldElement::<E>::zero(); domain_size];
    {
        let out_ptr = out_vec.as_mut_ptr() as *mut u64;
        let out_slice = unsafe { core::slice::from_raw_parts_mut(out_ptr, 3 * domain_size) };
        let mut outputs: [&mut [u64]; 1] = [out_slice];
        math_cuda::lde::coset_lde_batch_ext3_into(
            &inputs,
            n,
            blowup,
            &weights,
            &mut outputs,
        )
        .expect("GPU R4 deep-poly LDE failed");
    }
    Some(out_vec)
}

pub(crate) static GPU_R4_LDE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub fn gpu_r4_lde_calls() -> u64 {
    GPU_R4_LDE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// GPU path for the composition-polynomial LDE in the `number_of_parts > 2`
/// branch of `round_2_compute_composition_polynomial` (prover.rs:920). The
/// caller already has the polynomial parts; we batch their evaluations at
/// the `domain_size × blowup_factor` coset in a single GPU call.
///
/// Each part is padded to `domain_size` coefficients. Weights = `offset^k`
/// (coset shift, no 1/N normalisation — input is coefficients).
pub(crate) fn try_evaluate_parts_on_lde_gpu<F, E>(
    parts_coefs: &[&[FieldElement<E>]],
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
) -> Option<Vec<Vec<FieldElement<E>>>>
where
    F: math::field::traits::IsFFTField + IsField,
    E: IsField,
    F: IsSubFieldOf<E>,
{
    if parts_coefs.is_empty() {
        return Some(Vec::new());
    }
    if !domain_size.is_power_of_two() || !blowup_factor.is_power_of_two() {
        return None;
    }
    let lde_size = domain_size * blowup_factor;
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    let m = parts_coefs.len();

    GPU_PARTS_LDE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Weights: `offset^k` for k in 0..domain_size. F == Goldilocks.
    let mut weights_u64 = Vec::with_capacity(domain_size);
    let mut w = FieldElement::<F>::one();
    for _ in 0..domain_size {
        let v: u64 = unsafe { *(w.value() as *const _ as *const u64) };
        weights_u64.push(v);
        w = w * offset;
    }

    // Pack each part into a 3*domain_size u64 buffer, zero-padded.
    let mut part_bufs: Vec<Vec<u64>> = Vec::with_capacity(m);
    for part in parts_coefs.iter() {
        let mut buf = vec![0u64; 3 * domain_size];
        let len = part.len().min(domain_size);
        // Copy the real part coefficients; the rest stays zero (padding).
        let src_ptr = part.as_ptr() as *const u64;
        let src_len = len * 3;
        let src = unsafe { core::slice::from_raw_parts(src_ptr, src_len) };
        buf[..src_len].copy_from_slice(src);
        part_bufs.push(buf);
    }
    let input_slices: Vec<&[u64]> = part_bufs.iter().map(|v| v.as_slice()).collect();

    let mut outputs: Vec<Vec<FieldElement<E>>> = (0..m)
        .map(|_| vec![FieldElement::<E>::zero(); lde_size])
        .collect();
    {
        let mut out_slices: Vec<&mut [u64]> = outputs
            .iter_mut()
            .map(|o| {
                let ptr = o.as_mut_ptr() as *mut u64;
                unsafe { core::slice::from_raw_parts_mut(ptr, 3 * lde_size) }
            })
            .collect();
        math_cuda::lde::evaluate_poly_coset_batch_ext3_into(
            &input_slices,
            domain_size,
            blowup_factor,
            &weights_u64,
            &mut out_slices,
        )
        .expect("GPU parts LDE failed");
    }
    Some(outputs)
}

pub(crate) static GPU_PARTS_LDE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub fn gpu_parts_lde_calls() -> u64 {
    GPU_PARTS_LDE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Ext3 specialisation of [`try_expand_columns_batched`]. `E` is known to be
/// `Degree3GoldilocksExtensionField` by type_name match at the caller.
fn try_expand_columns_batched_ext3<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> bool
where
    F: IsField,
    E: IsField,
{
    if columns.is_empty() {
        return true;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);

    // SAFETY: caller confirmed `E == Degree3GoldilocksExtensionField` via
    // type_name. That means `FieldElement<E>` wraps `[FieldElement<Gl>; 3]`,
    // which is memory-equivalent to `[u64; 3]`. A `&[FieldElement<E>]` of
    // length `n` is therefore a contiguous `3 * n * 8` byte buffer.
    let raw_columns: Vec<Vec<u64>> = columns
        .iter()
        .map(|col| {
            let len = col.len() * 3;
            let ptr = col.as_ptr() as *const u64;
            // Copy rather than borrow: the caller still owns `col` and will
            // reuse its backing storage after we resize + rewrite below.
            unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
        })
        .collect();
    // F is `type_name::<F>() == GoldilocksField` by caller precondition;
    // `F::BaseType == u64`, so we can read each `w.value()` as a `*const u64`.
    let weights_u64: Vec<u64> = weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect();

    // Pre-size each ext3 column to lde_size so its backing Vec has the right
    // length for the output re-interleave. Capacity must already be >=
    // lde_size (caller's `extract_columns_main(lde_size)` ensures this).
    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        // SAFETY: overwritten fully by the GPU path below.
        unsafe { col.set_len(lde_size) };
    }

    // View each column's backing memory as a `&mut [u64]` of length
    // `3*lde_size`. Safe because ext3 elements are `[u64; 3]` layouts.
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len() * 3;
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();
    // Account each ext3 column as 3 logical GPU LDE "calls" (base-field
    // components) so the counter matches the base-field batched path.
    GPU_LDE_CALLS.fetch_add((columns.len() * 3) as u64, std::sync::atomic::Ordering::Relaxed);
    math_cuda::lde::coset_lde_batch_ext3_into(
        &slices,
        n,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
    )
    .expect("GPU batched ext3 coset LDE failed");
    true
}
