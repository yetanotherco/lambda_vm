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
