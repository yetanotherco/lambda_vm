//! GPU dispatch layer for the per-column coset LDE. Lives in the stark crate
//! (not `math`) to avoid a dependency cycle between `math` and `math-cuda`.
//!
//! Handles only Goldilocks base-field columns above a size threshold; falls
//! back to CPU for extension-field columns and small columns where kernel
//! launch overhead dominates. Produces the same natural-order, non-canonical
//! LDE evaluations as the CPU path.

use core::any::type_name;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    try_evaluate_parts_on_lde_gpu_impl(parts_coefs, blowup_factor, domain_size, offset, false)
        .map(|(v, _)| v)
}

/// Same as [`try_evaluate_parts_on_lde_gpu`] but also retains the
/// composition-parts LDE device buffer as a `GpuLdeExt3` handle. Used by
/// `round_2_compute_composition_polynomial` to feed R2 commit and R4
/// DEEP composition without re-H2D'ing.
pub(crate) fn try_evaluate_parts_on_lde_gpu_keep<F, E>(
    parts_coefs: &[&[FieldElement<E>]],
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
) -> Option<(Vec<Vec<FieldElement<E>>>, math_cuda::lde::GpuLdeExt3)>
where
    F: math::field::traits::IsFFTField + IsField,
    E: IsField,
    F: IsSubFieldOf<E>,
{
    let (v, h) = try_evaluate_parts_on_lde_gpu_impl(
        parts_coefs,
        blowup_factor,
        domain_size,
        offset,
        true,
    )?;
    Some((v, h.expect("keep=true returns Some handle")))
}

fn try_evaluate_parts_on_lde_gpu_impl<F, E>(
    parts_coefs: &[&[FieldElement<E>]],
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
    keep: bool,
) -> Option<(
    Vec<Vec<FieldElement<E>>>,
    Option<math_cuda::lde::GpuLdeExt3>,
)>
where
    F: math::field::traits::IsFFTField + IsField,
    E: IsField,
    F: IsSubFieldOf<E>,
{
    if parts_coefs.is_empty() {
        return Some((Vec::new(), None));
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

    let mut part_bufs: Vec<Vec<u64>> = Vec::with_capacity(m);
    for part in parts_coefs.iter() {
        let mut buf = vec![0u64; 3 * domain_size];
        let len = part.len().min(domain_size);
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
    let handle = {
        let mut out_slices: Vec<&mut [u64]> = outputs
            .iter_mut()
            .map(|o| {
                let ptr = o.as_mut_ptr() as *mut u64;
                unsafe { core::slice::from_raw_parts_mut(ptr, 3 * lde_size) }
            })
            .collect();
        if keep {
            Some(
                math_cuda::lde::evaluate_poly_coset_batch_ext3_into_keep(
                    &input_slices,
                    domain_size,
                    blowup_factor,
                    &weights_u64,
                    &mut out_slices,
                )
                .expect("GPU parts LDE (keep) failed"),
            )
        } else {
            math_cuda::lde::evaluate_poly_coset_batch_ext3_into(
                &input_slices,
                domain_size,
                blowup_factor,
                &weights_u64,
                &mut out_slices,
            )
            .expect("GPU parts LDE failed");
            None
        }
    };
    Some((outputs, handle))
}

/// Fused variant of [`try_evaluate_parts_on_lde_gpu`]: in addition to the
/// LDE parts, builds the R2 composition-polynomial Merkle tree on device
/// (row-pair Keccak leaves + pair-hash inner tree). Returns both the parts
/// (still needed downstream for R4 openings) and the finished tree.
#[allow(dead_code)]
pub(crate) fn try_evaluate_parts_on_lde_and_commit_gpu<F, E, B>(
    parts_coefs: &[&[FieldElement<E>]],
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
) -> Option<(
    Vec<Vec<FieldElement<E>>>,
    crypto::merkle_tree::merkle::MerkleTree<B>,
)>
where
    F: math::field::traits::IsFFTField + IsField,
    E: IsField,
    F: IsSubFieldOf<E>,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if parts_coefs.is_empty() {
        return None;
    }
    if !domain_size.is_power_of_two() || !blowup_factor.is_power_of_two() {
        return None;
    }
    let lde_size = domain_size * blowup_factor;
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if lde_size < 2 {
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
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Weights: `offset^k`.
    let mut weights_u64 = Vec::with_capacity(domain_size);
    let mut w = FieldElement::<F>::one();
    for _ in 0..domain_size {
        let v: u64 = unsafe { *(w.value() as *const _ as *const u64) };
        weights_u64.push(v);
        w = w * offset;
    }

    // Pack parts into per-part 3*domain_size u64 buffers (zero-padded).
    let mut part_bufs: Vec<Vec<u64>> = Vec::with_capacity(m);
    for part in parts_coefs.iter() {
        let mut buf = vec![0u64; 3 * domain_size];
        let len = part.len().min(domain_size);
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
    let num_leaves = lde_size / 2;
    let tight_total_nodes = 2 * num_leaves - 1;
    let mut nodes_bytes = vec![0u8; tight_total_nodes * 32];
    {
        let mut out_slices: Vec<&mut [u64]> = outputs
            .iter_mut()
            .map(|o| {
                let ptr = o.as_mut_ptr() as *mut u64;
                unsafe { core::slice::from_raw_parts_mut(ptr, 3 * lde_size) }
            })
            .collect();
        math_cuda::lde::evaluate_poly_coset_batch_ext3_into_with_merkle_tree(
            &input_slices,
            domain_size,
            blowup_factor,
            &weights_u64,
            &mut out_slices,
            &mut nodes_bytes,
        )
        .expect("GPU ext3 evaluate+commit failed");
    }

    // Build the MerkleTree from the device-produced nodes.
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(tight_total_nodes);
    for i in 0..tight_total_nodes {
        let mut n = [0u8; 32];
        n.copy_from_slice(&nodes_bytes[i * 32..(i + 1) * 32]);
        nodes.push(n);
    }
    let tree = crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)?;
    Some((outputs, tree))
}

/// Build a FRI-layer Merkle tree from already-folded evaluations using the
/// GPU pair-leaf kernel + pair-hash inner tree.
///
/// Not currently wired — benchmarking showed the win per layer (GPU tree
/// vs rayon tree) is eaten by the H2D of each layer's eval slab since the
/// evals are in pageable CPU Vec form at call time. A fused on-device FRI
/// (fold + leaves + tree all staying on device across layers) would flip
/// this but is deferred to the "LDE on GPU across rounds" item.
#[allow(dead_code)]
pub(crate) fn try_build_fri_layer_tree_gpu<E, B>(
    evals: &[FieldElement<E>],
) -> Option<crypto::merkle_tree::merkle::MerkleTree<B>>
where
    E: IsField,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let num_evals = evals.len();
    if num_evals < 2 || !num_evals.is_power_of_two() {
        return None;
    }
    let num_leaves = num_evals / 2;
    // Higher threshold than the generic LDE path because each FRI layer
    // H2Ds a fresh eval slab; tiny layers can't amortise that.
    if num_leaves < gpu_fri_tree_threshold() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }

    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // SAFETY: E == Ext3 whose BaseType is [FieldElement<Gl>; 3] =
    // contiguous [u64; 3] at runtime.
    let evals_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, num_evals * 3) };
    let nodes_bytes = math_cuda::merkle::build_fri_layer_tree_from_evals_ext3(evals_raw)
        .expect("GPU FRI layer tree build failed");

    let tight_total_nodes = 2 * num_leaves - 1;
    debug_assert_eq!(nodes_bytes.len(), tight_total_nodes * 32);
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(tight_total_nodes);
    for i in 0..tight_total_nodes {
        let mut n = [0u8; 32];
        n.copy_from_slice(&nodes_bytes[i * 32..(i + 1) * 32]);
        nodes.push(n);
    }
    crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)
}

/// Build the R2 composition-polynomial Merkle tree from already-computed
/// LDE parts using the GPU row-pair leaf kernel + pair-hash inner tree.
/// Takes H2D for every call — only worth doing when the tree is large enough
/// that CPU rayon Merkle build exceeds the round-trip cost.
pub(crate) fn try_build_comp_poly_tree_gpu<E, B>(
    lde_parts: &[Vec<FieldElement<E>>],
) -> Option<crypto::merkle_tree::merkle::MerkleTree<B>>
where
    E: IsField,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if lde_parts.is_empty() {
        return None;
    }
    let lde_size = lde_parts[0].len();
    if !lde_size.is_power_of_two() || lde_size < 2 {
        return None;
    }
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    // All parts same length.
    if lde_parts.iter().any(|p| p.len() != lde_size) {
        return None;
    }

    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // SAFETY: E == Ext3 whose BaseType is [FieldElement<Gl>; 3] =
    // contiguous [u64; 3] at runtime.
    let raw_parts: Vec<&[u64]> = lde_parts
        .iter()
        .map(|p| unsafe { core::slice::from_raw_parts(p.as_ptr() as *const u64, p.len() * 3) })
        .collect();

    let nodes_bytes = math_cuda::merkle::build_comp_poly_tree_from_evals_ext3(&raw_parts)
        .expect("GPU comp-poly tree build failed");

    let num_leaves = lde_size / 2;
    let tight_total_nodes = 2 * num_leaves - 1;
    debug_assert_eq!(nodes_bytes.len(), tight_total_nodes * 32);
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(tight_total_nodes);
    for i in 0..tight_total_nodes {
        let mut n = [0u8; 32];
        n.copy_from_slice(&nodes_bytes[i * 32..(i + 1) * 32]);
        nodes.push(n);
    }
    crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)
}

pub(crate) static GPU_PARTS_LDE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub fn gpu_parts_lde_calls() -> u64 {
    GPU_PARTS_LDE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Combined GPU LDE + Merkle leaf hash for the base-field main trace.
///
/// Keeps LDE output on device, runs Keccak-256 on the device buffer directly,
/// D2Hs both LDE columns (for Round 2-4 reuse) and hashed leaves (for tree
/// construction). Avoids the second H2D that a separate GPU Merkle commit
/// path would require.
///
/// On success: resizes each `columns[c]` to `lde_size` with the LDE output,
/// and returns `Vec<Commitment>` — the Keccak-256 hashed leaves in natural
/// row order, ready to pass to `BatchedMerkleTree::build_from_hashed_leaves`.
#[allow(dead_code)]
pub(crate) fn try_expand_and_leaf_hash_batched<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<Vec<[u8; 32]>>
where
    F: IsField,
    E: IsField,
{
    if columns.is_empty() {
        return Some(Vec::new());
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<GoldilocksField>() {
        return None;
    }
    if columns.iter().any(|c| c.len() != n) {
        return None;
    }

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

    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len();
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    // Allocate as Vec<[u8; 32]> directly so we both skip the zero-fill pass
    // AND avoid re-chunking afterwards. Fresh pages still fault on first
    // write (inside the GPU-side memcpy), but only once each.
    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(lde_size);
    // SAFETY: we fill every byte via memcpy_dtoh below.
    unsafe { leaves.set_len(lde_size) };
    let hashed_bytes_ptr = leaves.as_mut_ptr() as *mut u8;
    let hashed_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(hashed_bytes_ptr, lde_size * 32) };

    GPU_LDE_CALLS.fetch_add(columns.len() as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    math_cuda::lde::coset_lde_batch_base_into_with_leaf_hash(
        &slices,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
        hashed_bytes,
    )
    .expect("GPU LDE+leaf-hash failed");

    Some(leaves)
}

pub(crate) static GPU_LEAF_HASH_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub fn gpu_leaf_hash_calls() -> u64 {
    GPU_LEAF_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fused variant: LDE + leaf-hash + Merkle tree build, all on device. Skips
/// the pinned→pageable→pinned leaf dance of the separate-step pipeline.
/// Returns the filled `MerkleTree<B>` alongside populating `columns` with
/// the LDE-expanded evaluations.
#[allow(dead_code)]
pub(crate) fn try_expand_leaf_and_tree_batched<F, E, B>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<crypto::merkle_tree::merkle::MerkleTree<B>>
where
    F: IsField,
    E: IsField,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if columns.is_empty() {
        return None;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<GoldilocksField>() {
        return None;
    }
    if columns.iter().any(|c| c.len() != n) {
        return None;
    }
    // Tree layout needs `2*lde_size - 1` nodes; must be a power-of-two leaf
    // count. LDE size is always pow2 here (checked above).
    if lde_size < 2 {
        return None;
    }

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

    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len();
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    let total_nodes = 2 * lde_size - 1;
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    // SAFETY: every byte is written by the D2H below.
    unsafe { nodes.set_len(total_nodes) };
    let nodes_bytes: &mut [u8] = unsafe {
        core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, total_nodes * 32)
    };

    GPU_LDE_CALLS.fetch_add(columns.len() as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    math_cuda::lde::coset_lde_batch_base_into_with_merkle_tree(
        &slices,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
        nodes_bytes,
    )
    .expect("GPU LDE+leaf-hash+tree failed");

    crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)
}

/// Same as [`try_expand_leaf_and_tree_batched`] but ALSO retains the LDE
/// device buffer so R2–R4 GPU paths can reuse the LDE without a re-H2D.
/// Returns `(tree, gpu_handle)` on success, `None` if the GPU path doesn't
/// apply (same gates as the non-`_keep` variant).
pub(crate) fn try_expand_leaf_and_tree_batched_keep<F, E, B>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<(
    crypto::merkle_tree::merkle::MerkleTree<B>,
    math_cuda::lde::GpuLdeBase,
)>
where
    F: IsField,
    E: IsField,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if columns.is_empty() {
        return None;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<GoldilocksField>() {
        return None;
    }
    if columns.iter().any(|c| c.len() != n) {
        return None;
    }
    if lde_size < 2 {
        return None;
    }

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

    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len();
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    let total_nodes = 2 * lde_size - 1;
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    unsafe { nodes.set_len(total_nodes) };
    let nodes_bytes: &mut [u8] = unsafe {
        core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, total_nodes * 32)
    };

    GPU_LDE_CALLS.fetch_add(columns.len() as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let handle = math_cuda::lde::coset_lde_batch_base_into_with_merkle_tree_keep(
        &slices,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
        nodes_bytes,
    )
    .expect("GPU LDE+leaf-hash+tree+keep failed");

    let tree = crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)?;
    Some((tree, handle))
}

/// Ext3 variant of [`try_expand_leaf_and_tree_batched`]. Same fused flow
/// (LDE → leaf-hash → tree build) but over ext3 columns via the three-slab
/// decomposition; `B::Node = [u8; 32]` by construction for
/// `BatchKeccak256Backend<Ext3>`.
#[allow(dead_code)]
pub(crate) fn try_expand_leaf_and_tree_batched_ext3<F, E, B>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<crypto::merkle_tree::merkle::MerkleTree<B>>
where
    F: IsField,
    E: IsField,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if columns.is_empty() {
        return None;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if lde_size < 2 {
        return None;
    }

    // SAFETY: `E == Degree3Goldilocks`; each `FieldElement<E>` is
    // memory-equivalent to `[u64; 3]`. Copy out a Vec<u64> view per column.
    let raw_columns: Vec<Vec<u64>> = columns
        .iter()
        .map(|col| {
            let len = col.len() * 3;
            let ptr = col.as_ptr() as *const u64;
            unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
        })
        .collect();
    let weights_u64: Vec<u64> = weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect();

    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len() * 3;
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    let total_nodes = 2 * lde_size - 1;
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    unsafe { nodes.set_len(total_nodes) };
    let nodes_bytes: &mut [u8] = unsafe {
        core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, total_nodes * 32)
    };

    GPU_LDE_CALLS.fetch_add((columns.len() * 3) as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    math_cuda::lde::coset_lde_batch_ext3_into_with_merkle_tree(
        &slices,
        n,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
        nodes_bytes,
    )
    .expect("GPU ext3 LDE+leaf-hash+tree failed");

    crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)
}

/// Same as [`try_expand_leaf_and_tree_batched_ext3`] but also returns the
/// ext3 LDE device buffer (de-interleaved 3-slab layout) so downstream GPU
/// rounds can reuse it.
pub(crate) fn try_expand_leaf_and_tree_batched_ext3_keep<F, E, B>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<(
    crypto::merkle_tree::merkle::MerkleTree<B>,
    math_cuda::lde::GpuLdeExt3,
)>
where
    F: IsField,
    E: IsField,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if columns.is_empty() {
        return None;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if lde_size < 2 {
        return None;
    }

    let raw_columns: Vec<Vec<u64>> = columns
        .iter()
        .map(|col| {
            let len = col.len() * 3;
            let ptr = col.as_ptr() as *const u64;
            unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
        })
        .collect();
    let weights_u64: Vec<u64> = weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect();

    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len() * 3;
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    let total_nodes = 2 * lde_size - 1;
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    unsafe { nodes.set_len(total_nodes) };
    let nodes_bytes: &mut [u8] = unsafe {
        core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, total_nodes * 32)
    };

    GPU_LDE_CALLS.fetch_add((columns.len() * 3) as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let handle = math_cuda::lde::coset_lde_batch_ext3_into_with_merkle_tree_keep(
        &slices,
        n,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
        nodes_bytes,
    )
    .expect("GPU ext3 LDE+leaf-hash+tree+keep failed");

    let tree = crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)?;
    Some((tree, handle))
}

/// Ext3 variant of [`try_expand_and_leaf_hash_batched`] for the aux trace.
/// Decomposes each ext3 column into three base slabs, runs the LDE + Keccak
/// ext3 kernel in one on-device pipeline, re-interleaves LDE output back to
/// ext3 layout, and returns hashed leaves.
#[allow(dead_code)]
pub(crate) fn try_expand_and_leaf_hash_batched_ext3<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<Vec<[u8; 32]>>
where
    F: IsField,
    E: IsField,
{
    if columns.is_empty() {
        return Some(Vec::new());
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if columns.iter().any(|c| c.len() != n) {
        return None;
    }

    let raw_columns: Vec<Vec<u64>> = columns
        .iter()
        .map(|col| {
            let len = col.len() * 3;
            let ptr = col.as_ptr() as *const u64;
            unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
        })
        .collect();
    let weights_u64: Vec<u64> = weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect();

    for col in columns.iter_mut() {
        debug_assert!(col.capacity() >= lde_size);
        unsafe { col.set_len(lde_size) };
    }
    let mut raw_outputs: Vec<&mut [u64]> = columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len() * 3;
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect();

    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(lde_size);
    unsafe { leaves.set_len(lde_size) };
    let hashed_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(leaves.as_mut_ptr() as *mut u8, lde_size * 32)
    };

    GPU_LDE_CALLS.fetch_add((columns.len() * 3) as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    math_cuda::lde::coset_lde_batch_ext3_into_with_leaf_hash(
        &slices,
        n,
        blowup_factor,
        &weights_u64,
        &mut raw_outputs,
        hashed_bytes,
    )
    .expect("GPU ext3 LDE+leaf-hash failed");

    Some(leaves)
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

// ============================================================================
// GPU barycentric OOD evaluation
// ============================================================================
//
// Infrastructure for future use: these wrappers drive
// `math_cuda::barycentric::barycentric_{base,ext3}` and apply the trailing ext3
// scalar on host. See the CPU reference in
// `crypto/math/src/polynomial/mod.rs::interpolate_coset_eval_*_with_g_n_inv`.
//
// NOT currently wired into the prover — a benchmark on fib_iterative_{1M, 4M}
// showed the CPU path (rayon over ~50 columns) already finishes in <1 ms wall
// because the GPU is busy with LDE and Merkle on parallel streams, so moving
// R3 OOD to the GPU just serialises work without freeing CPU wall time.
// Kept here and covered by parity tests in `crypto/math-cuda/tests/barycentric.rs`
// because it remains a net win for single-table or very-large-trace workloads.
//
// The GPU kernel returns the unscaled sum
//     S = Σ_i point_i · eval_i · inv_denom_i
// per column; the final barycentric value is
//     f(z) = scalar · (z^N − g^N) · S
// with `scalar = n_inv · g_n_inv` kept in the base field.

static GPU_BARY_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn gpu_bary_calls() -> u64 {
    GPU_BARY_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

static GPU_DEEP_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn gpu_deep_calls() -> u64 {
    GPU_DEEP_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

static GPU_FRI_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn gpu_fri_calls() -> u64 {
    GPU_FRI_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// GPU-resident FRI commit phase. Keeps evals, twiddles, and per-layer
/// trees on device across all folds. Mirrors
/// `commit_phase_from_evaluations` on CPU (transcript interleaving
/// unchanged — each layer's zeta is sampled from the host transcript,
/// each layer's root is D2H'd and appended there).
///
/// Returns `None` to fall back to CPU (small domain, type mismatch, etc.).
#[allow(clippy::type_complexity)]
pub(crate) fn try_fri_commit_gpu<F, E>(
    number_layers: usize,
    evals: &[FieldElement<E>],
    transcript: &mut impl crypto::fiat_shamir::is_transcript::IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> Option<(
    FieldElement<E>,
    Vec<crate::fri::fri_commitment::FriLayer<E, crate::config::FriLayerMerkleTreeBackend<E>>>,
)>
where
    F: math::field::traits::IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: math::traits::AsBytes + Sync + Send,
    FieldElement<E>: math::traits::AsBytes + Sync + Send,
{
    use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
    use math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset;

    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if !domain_size.is_power_of_two() || domain_size < gpu_lde_threshold() {
        return None;
    }
    if evals.len() != domain_size || number_layers < 1 {
        return None;
    }
    if domain_size < (1 << 3) {
        return None;
    }

    GPU_FRI_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Compute initial inv_twiddles on host — same recipe as
    // `compute_coset_twiddles_inv`.
    let half = domain_size / 2;
    let order = domain_size.trailing_zeros() as u64;
    let mut points = get_powers_of_primitive_root_coset(order, half, coset_offset)
        .expect("coset twiddles available");
    in_place_bit_reverse_permute(&mut points);
    FieldElement::inplace_batch_inverse(&mut points).expect("twiddle inverse");

    // Raw u64 views: E == Ext3 (3 u64) for evals, F == Gl (1 u64) for twiddles.
    let evals_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, domain_size * 3) };
    let tw_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(points.as_ptr() as *const u64, half) };

    let mut state = math_cuda::fri::FriCommitState::new(evals_raw, tw_raw, domain_size)
        .expect("FRI state alloc");

    let mut fri_layer_list =
        Vec::<crate::fri::fri_commitment::FriLayer<E, crate::config::FriLayerMerkleTreeBackend<E>>>::with_capacity(number_layers);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;

    for _ in 1..number_layers {
        let zeta: FieldElement<E> = transcript.sample_field_element();
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;

        // SAFETY: E == Ext3 (layout [u64; 3]).
        let zeta_raw: [u64; 3] = unsafe {
            let p = &zeta as *const FieldElement<E> as *const u64;
            [*p, *p.add(1), *p.add(2)]
        };

        let (root_bytes, layer_evals_raw, nodes_bytes) =
            state.fold_and_commit_layer(zeta_raw).expect("FRI fold+commit");

        let mut root_arr = [0u8; 32];
        root_arr.copy_from_slice(&root_bytes[..32]);

        // Re-chunk tree nodes into Vec<[u8; 32]> for MerkleTree.
        let num_leaves = current_domain_size / 2;
        let tight_total_nodes = 2 * num_leaves - 1;
        debug_assert_eq!(nodes_bytes.len(), tight_total_nodes * 32);
        let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(tight_total_nodes);
        for i in 0..tight_total_nodes {
            let mut n = [0u8; 32];
            n.copy_from_slice(&nodes_bytes[i * 32..(i + 1) * 32]);
            nodes.push(n);
        }
        let merkle_tree =
            crypto::merkle_tree::merkle::MerkleTree::<crate::config::FriLayerMerkleTreeBackend<E>>::from_precomputed_nodes(nodes)
                .expect("FRI MerkleTree build");

        // Rebuild the layer's ext3 evals from raw u64s.
        debug_assert_eq!(layer_evals_raw.len(), 3 * current_domain_size);
        let mut layer_evals: Vec<FieldElement<E>> = Vec::with_capacity(current_domain_size);
        unsafe { layer_evals.set_len(current_domain_size) };
        unsafe {
            core::ptr::copy_nonoverlapping(
                layer_evals_raw.as_ptr(),
                layer_evals.as_mut_ptr() as *mut u64,
                current_domain_size * 3,
            );
        }

        fri_layer_list.push(crate::fri::fri_commitment::FriLayer::new(
            &layer_evals,
            merkle_tree,
            current_coset_offset.clone().to_extension(),
            current_domain_size,
        ));

        transcript.append_bytes(&root_arr);
    }

    // Final fold.
    let zeta: FieldElement<E> = transcript.sample_field_element();
    let zeta_raw: [u64; 3] = unsafe {
        let p = &zeta as *const FieldElement<E> as *const u64;
        [*p, *p.add(1), *p.add(2)]
    };
    let last_raw = state.fold_final(zeta_raw).expect("FRI final fold");

    // SAFETY: E == Ext3; build FieldElement<E> from raw u64s.
    let last_value: FieldElement<E> = unsafe {
        let mut e: FieldElement<E> = core::mem::zeroed();
        let ptr = &mut e as *mut FieldElement<E> as *mut u64;
        *ptr = last_raw[0];
        *ptr.add(1) = last_raw[1];
        *ptr.add(2) = last_raw[2];
        e
    };

    transcript.append_field_element(&last_value);

    Some((last_value, fri_layer_list))
}

/// R3 OOD barycentric over the **main** (base-field) LDE read directly from
/// the device handle with stride `row_stride = blowup_factor`. Applies the
/// same trailing `scalar * vanishing * sum` ext3 scale on host that
/// `interpolate_coset_eval_with_g_n_inv` does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_base_on_handle<F, E>(
    lde_trace: &crate::trace::LDETraceTable<F, E>,
    row_stride: usize,
    coset_points: &[FieldElement<F>],
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
    inv_denoms: &[FieldElement<E>],
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let main = lde_trace.gpu_main()?;
    let num_cols = main.m;
    if num_cols == 0 {
        return Some(Vec::new());
    }
    let n = coset_points.len();
    if !n.is_power_of_two() || n < gpu_bary_threshold() {
        return None;
    }
    if inv_denoms.len() != n || main.lde_size != n * row_stride {
        return None;
    }

    GPU_BARY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let points_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(coset_points.as_ptr() as *const u64, n) };
    let inv_denoms_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(inv_denoms.as_ptr() as *const u64, 3 * n) };

    let sums_raw = math_cuda::barycentric::barycentric_base_on_device(
        main,
        row_stride,
        points_raw,
        inv_denoms_raw,
        n,
    )
    .expect("GPU barycentric_base_on_device failed");

    let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
    Some(apply_ext3_scalar::<E>(&sums_raw, scalar, num_cols))
}

/// Ext3 counterpart reading the aux LDE handle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_ext3_on_handle<F, E>(
    lde_trace: &crate::trace::LDETraceTable<F, E>,
    row_stride: usize,
    coset_points: &[FieldElement<F>],
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
    inv_denoms: &[FieldElement<E>],
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let aux = lde_trace.gpu_aux()?;
    let num_cols = aux.m;
    if num_cols == 0 {
        return Some(Vec::new());
    }
    let n = coset_points.len();
    if !n.is_power_of_two() || n < gpu_bary_threshold() {
        return None;
    }
    if inv_denoms.len() != n || aux.lde_size != n * row_stride {
        return None;
    }

    GPU_BARY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let points_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(coset_points.as_ptr() as *const u64, n) };
    let inv_denoms_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(inv_denoms.as_ptr() as *const u64, 3 * n) };

    let sums_raw = math_cuda::barycentric::barycentric_ext3_on_device(
        aux,
        row_stride,
        points_raw,
        inv_denoms_raw,
        n,
    )
    .expect("GPU barycentric_ext3_on_device failed");

    let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
    Some(apply_ext3_scalar::<E>(&sums_raw, scalar, num_cols))
}

/// GPU path for `compute_deep_composition_poly_evaluations`. Returns the N
/// trace-size coset evaluations of the deep-composition polynomial as a
/// `Vec<FieldElement<E>>` (same type as the CPU path), or `None` when the
/// GPU is skipped (small tables, handle absent, type mismatch).
///
/// Reads the main/aux LDE from the device handles stored on the
/// `LDETraceTable` by R1, avoiding a re-H2D of the largest tensor in R4.
/// Composition-parts LDE + scalar arrays are still H2D'd fresh each call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_deep_composition_gpu<F, E>(
    lde_trace: &crate::trace::LDETraceTable<F, E>,
    h_lde_parts: &[Vec<FieldElement<E>>],
    h_parts_gpu: Option<&math_cuda::lde::GpuLdeExt3>,
    h_ood: &[FieldElement<E>],
    trace_ood_cols: &[Vec<FieldElement<E>>], // num_total_cols × num_eval_points
    gammas_h: &[FieldElement<E>],
    gammas_tr_flat: &[Vec<FieldElement<E>>], // num_total_cols × num_eval_points
    inv_h: &[FieldElement<E>],
    inv_t: &[Vec<FieldElement<E>>], // num_eval_points × domain_size
    num_eval_points: usize,
    blowup_factor: usize,
    domain_size: usize,
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }

    let main_handle = lde_trace.gpu_main()?.clone();
    let aux_handle_opt = lde_trace.gpu_aux().cloned();
    let num_main = main_handle.m;
    let lde_size = main_handle.lde_size;
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    let num_aux = aux_handle_opt.as_ref().map(|a| a.m).unwrap_or(0);
    let num_parts = h_lde_parts.len();
    let num_total_cols = num_main + num_aux;

    if h_lde_parts.iter().any(|p| p.len() != lde_size) {
        return None;
    }

    GPU_DEEP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // If a device handle is present for h_parts, skip the host-side pack.
    // Falls back to packing Vec<Vec<_>> → flat u64 and H2D'ing in the
    // impl otherwise.
    let h_flat_opt: Option<Vec<u64>> = if h_parts_gpu.is_some() {
        None
    } else {
        let mut h_flat = vec![0u64; num_parts * 3 * lde_size];
        #[cfg(feature = "parallel")]
        let iter = h_lde_parts.par_iter().enumerate();
        #[cfg(not(feature = "parallel"))]
        let iter = h_lde_parts.iter().enumerate();
        let ptr = h_flat.as_mut_ptr() as usize;
        iter.for_each(|(p, col)| {
            // SAFETY: E == Ext3; FieldElement<E> is [u64; 3] at runtime.
            let src = unsafe { core::slice::from_raw_parts(col.as_ptr() as *const u64, lde_size * 3) };
            unsafe {
                let base = ptr as *mut u64;
                let slab0 = base.add((p * 3) * lde_size);
                let slab1 = base.add((p * 3 + 1) * lde_size);
                let slab2 = base.add((p * 3 + 2) * lde_size);
                for r in 0..lde_size {
                    *slab0.add(r) = src[r * 3];
                    *slab1.add(r) = src[r * 3 + 1];
                    *slab2.add(r) = src[r * 3 + 2];
                }
            }
        });
        Some(h_flat)
    };

    // Pack scalar arrays: h_ood, trace_ood, gammas_h, gammas_tr, inv_h, inv_t.
    let e3_raw = |e: &FieldElement<E>| -> [u64; 3] {
        // SAFETY: E == Ext3; memory layout [u64; 3].
        unsafe {
            let p = e as *const FieldElement<E> as *const u64;
            [*p, *p.add(1), *p.add(2)]
        }
    };

    let mut h_ood_flat = vec![0u64; num_parts * 3];
    for (j, e) in h_ood.iter().enumerate() {
        let v = e3_raw(e);
        h_ood_flat[j * 3] = v[0];
        h_ood_flat[j * 3 + 1] = v[1];
        h_ood_flat[j * 3 + 2] = v[2];
    }
    assert_eq!(trace_ood_cols.len(), num_total_cols);
    let mut trace_ood_flat = vec![0u64; num_total_cols * num_eval_points * 3];
    for (j, col) in trace_ood_cols.iter().enumerate() {
        debug_assert_eq!(col.len(), num_eval_points);
        for (k, e) in col.iter().enumerate() {
            let v = e3_raw(e);
            let idx = (j * num_eval_points + k) * 3;
            trace_ood_flat[idx] = v[0];
            trace_ood_flat[idx + 1] = v[1];
            trace_ood_flat[idx + 2] = v[2];
        }
    }
    let mut gammas_h_flat = vec![0u64; num_parts * 3];
    for (j, e) in gammas_h.iter().enumerate() {
        let v = e3_raw(e);
        gammas_h_flat[j * 3] = v[0];
        gammas_h_flat[j * 3 + 1] = v[1];
        gammas_h_flat[j * 3 + 2] = v[2];
    }
    assert_eq!(gammas_tr_flat.len(), num_total_cols);
    let mut gammas_tr_out = vec![0u64; num_total_cols * num_eval_points * 3];
    for (j, col) in gammas_tr_flat.iter().enumerate() {
        debug_assert_eq!(col.len(), num_eval_points);
        for (k, e) in col.iter().enumerate() {
            let v = e3_raw(e);
            let idx = (j * num_eval_points + k) * 3;
            gammas_tr_out[idx] = v[0];
            gammas_tr_out[idx + 1] = v[1];
            gammas_tr_out[idx + 2] = v[2];
        }
    }
    // SAFETY: E == Ext3; each FieldElement<E> is `[u64; 3]`. Cast the
    // contiguous Vec<FieldElement<E>> layer to a `&[u64]` and memcpy once,
    // instead of a per-element u64 copy loop.
    let inv_h_flat: Vec<u64> = unsafe {
        core::slice::from_raw_parts(inv_h.as_ptr() as *const u64, inv_h.len() * 3)
    }
    .to_vec();
    assert_eq!(inv_t.len(), num_eval_points);
    let mut inv_t_flat: Vec<u64> = Vec::with_capacity(num_eval_points * domain_size * 3);
    unsafe { inv_t_flat.set_len(num_eval_points * domain_size * 3) };
    {
        let dst_ptr = inv_t_flat.as_mut_ptr() as usize;
        #[cfg(feature = "parallel")]
        let iter = (0..num_eval_points).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..num_eval_points;
        iter.for_each(|k| {
            let layer = &inv_t[k];
            let src = unsafe {
                core::slice::from_raw_parts(layer.as_ptr() as *const u64, domain_size * 3)
            };
            unsafe {
                let dst = (dst_ptr as *mut u64).add(k * domain_size * 3);
                core::ptr::copy_nonoverlapping(src.as_ptr(), dst, domain_size * 3);
            }
        });
    }

    let raw_out = if let Some(h_gpu) = h_parts_gpu {
        math_cuda::deep::deep_composition_ext3_with_dev_parts(
            &main_handle,
            aux_handle_opt.as_ref(),
            h_gpu,
            &h_ood_flat,
            &trace_ood_flat,
            &gammas_h_flat,
            &gammas_tr_out,
            &inv_h_flat,
            &inv_t_flat,
            num_parts,
            num_main,
            num_aux,
            num_eval_points,
            blowup_factor,
            domain_size,
        )
        .expect("GPU deep composition (dev parts) failed")
    } else {
        math_cuda::deep::deep_composition_ext3(
            &main_handle,
            aux_handle_opt.as_ref(),
            h_flat_opt.as_ref().expect("host h_flat packed").as_slice(),
            &h_ood_flat,
            &trace_ood_flat,
            &gammas_h_flat,
            &gammas_tr_out,
            &inv_h_flat,
            &inv_t_flat,
            num_parts,
            num_main,
            num_aux,
            num_eval_points,
            blowup_factor,
            domain_size,
        )
        .expect("GPU deep composition failed")
    };

    // Transmute raw u64s → FieldElement<E>. Requires E == Ext3 layout, which
    // the type_name check above verifies.
    let mut out: Vec<FieldElement<E>> = Vec::with_capacity(domain_size);
    unsafe { out.set_len(domain_size) };
    let dst_ptr = out.as_mut_ptr() as *mut u64;
    unsafe {
        core::ptr::copy_nonoverlapping(raw_out.as_ptr(), dst_ptr, domain_size * 3);
    }
    Some(out)
}

// ============================================================================
// GPU Merkle inner-tree construction
// ============================================================================
//
// After the GPU keccak leaf-hash kernels produce a flat `[u8; 32]` leaf vec,
// the inner tree construction on CPU via `build_from_hashed_leaves` is a
// rayon-parallel pair-hash scan that still takes ~50-100 ms per table on a
// 46-core host. Delegating it to `math_cuda::merkle::build_merkle_tree_on_device`
// pushes it below 10 ms — the leaf buffer is already on host (it came out of
// `try_expand_and_leaf_hash_batched`), we H2D it once, the GPU does ~log₂(N)
// small kernel launches, and we D2H the full `2*leaves_len - 1` node array.

static GPU_MERKLE_TREE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn gpu_merkle_tree_calls() -> u64 {
    GPU_MERKLE_TREE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// FRI layers shrink by 2× each round; the last few layers are tiny. Below
/// this leaf count, keep the tree build on CPU.
#[allow(dead_code)]
const DEFAULT_GPU_FRI_TREE_THRESHOLD: usize = 1 << 19;

#[allow(dead_code)]
fn gpu_fri_tree_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_FRI_TREE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_FRI_TREE_THRESHOLD)
    })
}

/// Build a Merkle tree from already-hashed leaves using the GPU pair-hash
/// kernel. Returns the filled `MerkleTree` in the same layout as the CPU
/// `build_from_hashed_leaves` would produce — plug straight in anywhere the
/// prover expected that.
///
/// Returns `None` if the GPU path is disabled by threshold (`leaves_len <
/// GPU_MERKLE_TREE_THRESHOLD`), falling back to the caller's CPU path.
///
/// Currently unwired in the prover: benchmarking showed the savings from
/// the GPU pair-hash are eaten by the H2D of leaves + D2H of the tree
/// because the leaves are in pageable memory (they're the caller's Vec from
/// `try_expand_and_leaf_hash_batched`). A proper fusion would keep the
/// leaf buffer on device and run the tree kernel immediately on the GPU
/// copy — left as future work.
#[allow(dead_code)]
pub(crate) fn try_build_merkle_tree_gpu<B>(
    hashed_leaves: &[B::Node],
) -> Option<crypto::merkle_tree::merkle::MerkleTree<B>>
where
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let leaves_len = hashed_leaves.len();
    if leaves_len < gpu_merkle_tree_threshold() || !leaves_len.is_power_of_two() || leaves_len < 2 {
        return None;
    }
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Flatten host-side leaves into a contiguous byte buffer for the GPU
    // kernel. SAFETY: `[u8; 32]` is POD and the slice is contiguous.
    let leaves_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(hashed_leaves.as_ptr() as *const u8, leaves_len * 32)
    };
    let nodes_bytes = math_cuda::merkle::build_merkle_tree_on_device(leaves_bytes)
        .expect("GPU merkle tree build failed");

    let total_nodes = 2 * leaves_len - 1;
    debug_assert_eq!(nodes_bytes.len(), total_nodes * 32);

    // Re-chunk into `Vec<[u8; 32]>` without re-allocating. We'd need an
    // explicit copy because Vec<u8> and Vec<[u8; 32]> have different
    // layouts in the allocator metadata (align differs on some platforms).
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    for i in 0..total_nodes {
        let mut n = [0u8; 32];
        n.copy_from_slice(&nodes_bytes[i * 32..(i + 1) * 32]);
        nodes.push(n);
    }

    crypto::merkle_tree::merkle::MerkleTree::<B>::from_precomputed_nodes(nodes)
}

/// Below this (tree size), stay on CPU — rayon pair-hash is already well
/// under a millisecond for small N and would lose to any PCIe round-trip.
const DEFAULT_GPU_MERKLE_TREE_THRESHOLD: usize = 1 << 15;

fn gpu_merkle_tree_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_MERKLE_TREE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_MERKLE_TREE_THRESHOLD)
    })
}

/// Below this (trace-size) barycentric length we stay on CPU — the rayon path
/// already completes in well under a millisecond and PCIe round-trip would
/// dominate.
#[allow(dead_code)]
const DEFAULT_GPU_BARY_THRESHOLD: usize = 1 << 14;

#[allow(dead_code)]
fn gpu_bary_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_BARY_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_BARY_THRESHOLD)
    })
}

/// One ext3 scalar `(n_inv · g_n_inv) · (z^N − g^N)`; caller reads as `[u64;3]`.
#[allow(dead_code)]
fn ood_ext3_scalar<F, E>(
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
) -> [u64; 3]
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    // (z^N − g^N) in E — done via sub_subfield (E − F → E).
    let vanishing = z_pow_n.sub_subfield(coset_offset_pow_n);
    let base_scalar = n_inv * g_n_inv; // F × F → F
    let scalar_ext3: FieldElement<E> = &base_scalar * &vanishing; // F × E → E
    // SAFETY: E == Degree3Goldilocks; backing is `[FieldElement<Gl>; 3]`
    // which is memory-equivalent to `[u64; 3]`.
    let ptr = &scalar_ext3 as *const FieldElement<E> as *const u64;
    unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] }
}

/// Multiply each raw GPU ext3 sum by the host-computed ext3 scalar.
/// `sums_raw` is `3 * num_cols` u64s (interleaved).
#[allow(dead_code)]
fn apply_ext3_scalar<E>(
    sums_raw: &[u64],
    scalar: [u64; 3],
    num_cols: usize,
) -> Vec<FieldElement<E>>
where
    E: IsField,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    type Gl = GoldilocksField;
    type Ext3 = Degree3GoldilocksExtensionField;

    debug_assert_eq!(sums_raw.len(), 3 * num_cols);
    debug_assert_eq!(type_name::<E>(), type_name::<Ext3>());

    let scalar_e: FieldElement<Ext3> = FieldElement::<Ext3>::new([
        FieldElement::<Gl>::from_raw(scalar[0]),
        FieldElement::<Gl>::from_raw(scalar[1]),
        FieldElement::<Gl>::from_raw(scalar[2]),
    ]);

    let mut out: Vec<FieldElement<E>> = Vec::with_capacity(num_cols);
    for c in 0..num_cols {
        let s: FieldElement<Ext3> = FieldElement::<Ext3>::new([
            FieldElement::<Gl>::from_raw(sums_raw[c * 3]),
            FieldElement::<Gl>::from_raw(sums_raw[c * 3 + 1]),
            FieldElement::<Gl>::from_raw(sums_raw[c * 3 + 2]),
        ]);
        let final_ext3 = &s * &scalar_e;
        // SAFETY: E == Ext3 at runtime; same layout.
        let final_e: FieldElement<E> = unsafe {
            core::mem::transmute_copy::<FieldElement<Ext3>, FieldElement<E>>(&final_ext3)
        };
        out.push(final_e);
    }
    out
}

/// Batched barycentric OOD evaluation over M base-field columns at a single
/// ext3 evaluation point. Returns `Some(vec_of_M_ext3)` on GPU dispatch, or
/// `None` if the caller should fall back to CPU.
#[allow(dead_code)]
pub(crate) fn try_barycentric_base_ood_gpu<F, E>(
    columns: &[Vec<FieldElement<F>>],
    coset_points: &[FieldElement<F>],
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
    inv_denoms: &[FieldElement<E>],
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let num_cols = columns.len();
    if num_cols == 0 {
        return Some(Vec::new());
    }
    let n = columns[0].len();
    if !n.is_power_of_two() || n < gpu_bary_threshold() {
        return None;
    }
    if coset_points.len() != n || inv_denoms.len() != n {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    // All columns must share the same length `n`.
    for c in columns.iter() {
        if c.len() != n {
            return None;
        }
    }

    GPU_BARY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Pack columns contiguously: column c at offset c*n. Skip the zero-fill
    // prologue — we overwrite every byte below. `set_len` before write is
    // safe because `u64` has no drop glue.
    let total = num_cols * n;
    let mut columns_flat: Vec<u64> = Vec::with_capacity(total);
    unsafe { columns_flat.set_len(total) };
    {
        // Parallel pack: each column's slab is independent.
        let flat_ptr = columns_flat.as_mut_ptr() as usize;
        #[cfg(feature = "parallel")]
        let iter = (0..num_cols).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..num_cols;
        iter.for_each(|c| {
            // SAFETY: disjoint slabs; no two `c`s overlap. F == Goldilocks.
            unsafe {
                let dst = (flat_ptr as *mut u64).add(c * n);
                let src = columns[c].as_ptr() as *const u64;
                core::ptr::copy_nonoverlapping(src, dst, n);
            }
        });
    }
    let points_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(coset_points.as_ptr() as *const u64, n) };
    let inv_denoms_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(inv_denoms.as_ptr() as *const u64, 3 * n) };

    let sums_raw = math_cuda::barycentric::barycentric_base(
        &columns_flat,
        n,
        points_raw,
        inv_denoms_raw,
        n,
        num_cols,
    )
    .expect("GPU barycentric_base failed");

    let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
    Some(apply_ext3_scalar::<E>(&sums_raw, scalar, num_cols))
}

/// Batched barycentric OOD evaluation over M ext3 columns at a single ext3
/// evaluation point. Same contract as [`try_barycentric_base_ood_gpu`].
#[allow(dead_code)]
pub(crate) fn try_barycentric_ext3_ood_gpu<F, E>(
    columns: &[Vec<FieldElement<E>>],
    coset_points: &[FieldElement<F>],
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
    inv_denoms: &[FieldElement<E>],
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let num_cols = columns.len();
    if num_cols == 0 {
        return Some(Vec::new());
    }
    let n = columns[0].len();
    if !n.is_power_of_two() || n < gpu_bary_threshold() {
        return None;
    }
    if coset_points.len() != n || inv_denoms.len() != n {
        return None;
    }
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return None;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    for c in columns.iter() {
        if c.len() != n {
            return None;
        }
    }

    GPU_BARY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // De-interleaved layout: slab (c*3 + k) at offset (c*3+k)*n. Skip
    // zero-fill (we overwrite every byte). Parallelise the de-interleave.
    let total = num_cols * 3 * n;
    let mut columns_flat: Vec<u64> = Vec::with_capacity(total);
    unsafe { columns_flat.set_len(total) };
    {
        let flat_ptr = columns_flat.as_mut_ptr() as usize;
        #[cfg(feature = "parallel")]
        let iter = (0..num_cols).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..num_cols;
        iter.for_each(|c| {
            // SAFETY: E == Ext3 whose BaseType is [FieldElement<Gl>;3] =
            // contiguous [u64;3] at runtime; disjoint per-c slabs.
            unsafe {
                let src = columns[c].as_ptr() as *const u64;
                let base = flat_ptr as *mut u64;
                let slab0 = base.add((c * 3) * n);
                let slab1 = base.add((c * 3 + 1) * n);
                let slab2 = base.add((c * 3 + 2) * n);
                for r in 0..n {
                    *slab0.add(r) = *src.add(r * 3);
                    *slab1.add(r) = *src.add(r * 3 + 1);
                    *slab2.add(r) = *src.add(r * 3 + 2);
                }
            }
        });
    }
    let points_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(coset_points.as_ptr() as *const u64, n) };
    let inv_denoms_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(inv_denoms.as_ptr() as *const u64, 3 * n) };

    let sums_raw = math_cuda::barycentric::barycentric_ext3(
        &columns_flat,
        n,
        points_raw,
        inv_denoms_raw,
        n,
        num_cols,
    )
    .expect("GPU barycentric_ext3 failed");

    let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
    Some(apply_ext3_scalar::<E>(&sums_raw, scalar, num_cols))
}
