//! GPU dispatch layer for the per-column coset LDE.
//!
//! Handles only Goldilocks base-field columns above a size threshold; falls
//! back to CPU for extension-field columns and small columns where kernel
//! launch overhead dominates. Produces the same natural-order, non-canonical
//! LDE evaluations as the CPU path.

use std::any::TypeId;

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

/// Reset all GPU call counters at once. Useful between bench warm-up and
/// profiled passes so the numbers reported aren't doubled by the warm-up.
pub fn reset_all_gpu_call_counters() {
    use std::sync::atomic::Ordering::Relaxed;
    GPU_LDE_CALLS.store(0, Relaxed);
    GPU_EXTEND_HALVES_CALLS.store(0, Relaxed);
    GPU_LEAF_HASH_CALLS.store(0, Relaxed);
    GPU_MERKLE_TREE_CALLS.store(0, Relaxed);
}

pub(crate) static GPU_EXTEND_HALVES_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub fn gpu_extend_halves_calls() -> u64 {
    GPU_EXTEND_HALVES_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

// ============================================================================
// Shared dispatch helpers
// ============================================================================
//
// Every `try_expand_*` variant runs the same prologue: empty-check, threshold
// check, two TypeId checks, equal-length check, and a column-to-u64 cast under
// the type-confirmed precondition. Centralising that here keeps each variant
// short and means a future change to (say) the threshold logic is one edit.

/// Outcome of validating an incoming `columns` slice against the GPU dispatch
/// preconditions.
enum LayoutDispatch {
    /// `columns` is empty — caller returns its own "trivially done" value
    /// (`true` for `bool` callers, `Some(Vec::new())` for `Option` callers).
    Empty,
    /// GPU path doesn't apply (below threshold, wrong types, ragged columns).
    /// Caller returns its own "fall through to CPU" value (`false`/`None`).
    Skip,
    /// GPU path applies. `n` is the per-column input length; `lde_size = n *
    /// blowup_factor` (saturating).
    Run { n: usize, lde_size: usize },
}

/// Validate preconditions for the base-field batched GPU path: every column
/// must be Goldilocks base-field of equal length, the LDE size must clear the
/// threshold.
fn check_base_layout<F, E>(columns: &[Vec<FieldElement<E>>], blowup_factor: usize) -> LayoutDispatch
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if columns.is_empty() {
        return LayoutDispatch::Empty;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return LayoutDispatch::Skip;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return LayoutDispatch::Skip;
    }
    if TypeId::of::<E>() != TypeId::of::<GoldilocksField>() {
        return LayoutDispatch::Skip;
    }
    if columns.iter().any(|c| c.len() != n) {
        return LayoutDispatch::Skip;
    }
    LayoutDispatch::Run { n, lde_size }
}

/// Validate preconditions for the ext3 batched GPU path: every column must be
/// `Degree3GoldilocksExtensionField` of equal length, weights must be over
/// `GoldilocksField`, LDE size must clear the threshold.
fn check_ext3_layout<F, E>(columns: &[Vec<FieldElement<E>>], blowup_factor: usize) -> LayoutDispatch
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if columns.is_empty() {
        return LayoutDispatch::Empty;
    }
    let n = columns[0].len();
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return LayoutDispatch::Skip;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return LayoutDispatch::Skip;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return LayoutDispatch::Skip;
    }
    if columns.iter().any(|c| c.len() != n) {
        return LayoutDispatch::Skip;
    }
    LayoutDispatch::Run { n, lde_size }
}

/// Materialise base-field columns as owned `Vec<Vec<u64>>` for the GPU input
/// slice list.
///
/// SAFETY: caller must have established `E == GoldilocksField` (e.g. via
/// [`check_base_layout`]). Each `FieldElement<E>` is then a `#[repr(transparent)]`
/// wrapper over `u64`.
unsafe fn columns_to_u64_base<E: IsField>(columns: &[Vec<FieldElement<E>>]) -> Vec<Vec<u64>> {
    columns
        .iter()
        .map(|col| {
            col.iter()
                .map(|e| unsafe { *(e.value() as *const _ as *const u64) })
                .collect()
        })
        .collect()
}

/// Materialise ext3 columns as owned `Vec<Vec<u64>>` (de-interleaved into raw
/// `[u64; 3]` lanes per element) for the GPU input slice list.
///
/// SAFETY: caller must have established `E == Degree3GoldilocksExtensionField`
/// (e.g. via [`check_ext3_layout`]). Each `FieldElement<E>` is then a
/// `#[repr(transparent)]` wrapper over `[u64; 3]`.
unsafe fn columns_to_u64_ext3<E: IsField>(columns: &[Vec<FieldElement<E>>]) -> Vec<Vec<u64>> {
    columns
        .iter()
        .map(|col| {
            let len = col.len() * 3;
            let ptr = col.as_ptr() as *const u64;
            unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
        })
        .collect()
}

/// Materialise weights as a raw `Vec<u64>`.
///
/// SAFETY: caller must have established `F == GoldilocksField`.
unsafe fn weights_to_u64<F: IsField>(weights: &[FieldElement<F>]) -> Vec<u64> {
    weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect()
}

/// Pre-size each column to `lde_size` and view it as a `&mut [u64]` of length
/// `lde_size` (base-field, single-u64 layout). Asserts capacity hard so a
/// caller regression can't quietly UB in release builds.
///
/// SAFETY: caller must have established `E == GoldilocksField`.
unsafe fn presize_and_view_base<E: IsField>(
    columns: &mut [Vec<FieldElement<E>>],
    lde_size: usize,
) -> Vec<&mut [u64]> {
    for col in columns.iter_mut() {
        assert!(
            col.capacity() >= lde_size,
            "col capacity {} < lde_size {}",
            col.capacity(),
            lde_size
        );
        // SAFETY: assert above guarantees capacity; the GPU path overwrites
        // every slot before any reader sees the new length.
        unsafe { col.set_len(lde_size) };
    }
    columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len();
            // SAFETY: single-u64 layout, caller still owns the backing alloc.
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect()
}

/// Same as [`presize_and_view_base`] but for ext3 columns: each view is
/// `3 * lde_size` u64s (de-interleaved lanes).
///
/// SAFETY: caller must have established `E == Degree3GoldilocksExtensionField`.
unsafe fn presize_and_view_ext3<E: IsField>(
    columns: &mut [Vec<FieldElement<E>>],
    lde_size: usize,
) -> Vec<&mut [u64]> {
    for col in columns.iter_mut() {
        assert!(
            col.capacity() >= lde_size,
            "col capacity {} < lde_size {}",
            col.capacity(),
            lde_size
        );
        // SAFETY: assert above + GPU path overwrites every slot.
        unsafe { col.set_len(lde_size) };
    }
    columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len() * 3;
            // SAFETY: ext3 `[u64; 3]` layout, caller still owns the backing.
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        })
        .collect()
}

/// Truncate each column back to `n` (trace size) after a GPU error so the
/// CPU fallback (which reads `buffer.len()` as the trace size) runs cleanly.
/// Safe because `math_cuda` writes outputs only at the final host copy, post-
/// synchronize; any `Err` returns before that copy, leaving `columns[0..n]`
/// untouched.
fn restore_columns_on_err<E: IsField>(columns: &mut [Vec<FieldElement<E>>], n: usize) {
    for col in columns.iter_mut() {
        col.truncate(n);
    }
}

/// Allocate the `[u8; 32]` Merkle node buffer for a tree of `lde_size` leaves
/// and return both the node `Vec` (length-initialised, contents undefined) and
/// a `&mut [u8]` byte view of total length `total_nodes * 32`. Returns `None`
/// if the layout would be invalid (`lde_size < 2` or the byte length
/// overflows). The caller must overwrite every byte via the GPU D2H below.
fn alloc_merkle_nodes(lde_size: usize) -> Option<(Vec<[u8; 32]>, usize)> {
    if lde_size < 2 {
        return None;
    }
    let total_nodes = 2usize.saturating_mul(lde_size).checked_sub(1)?;
    let _byte_len = total_nodes.checked_mul(32)?;
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    // SAFETY: every byte will be overwritten via the GPU D2H before the
    // contents are read; the caller computes the byte-length view from the
    // returned `nodes` Vec using `total_nodes.checked_mul(32)`.
    unsafe { nodes.set_len(total_nodes) };
    Some((nodes, total_nodes))
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
    F: IsField + 'static,
    E: IsField + 'static,
{
    // Ext3 fast path: decompose each ext3 column into its 3 base components
    // and dispatch to the base-field batched NTT with 3×M logical columns.
    // Butterflies with a base-field twiddle act componentwise on ext3, so
    // this is exactly equivalent to running the NTT in the extension field.
    if TypeId::of::<E>() == TypeId::of::<Degree3GoldilocksExtensionField>() {
        return try_expand_columns_batched_ext3::<F, E>(columns, blowup_factor, weights);
    }

    let (n, lde_size) = match check_base_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty => return true, // nothing to do — same as CPU path
        LayoutDispatch::Skip => return false,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();

    // SAFETY: layout-checked above (`E == GoldilocksField`, `F == GoldilocksField`).
    let raw_columns = unsafe { columns_to_u64_base::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();
    GPU_LDE_CALLS.fetch_add(num_columns as u64, std::sync::atomic::Ordering::Relaxed);
    let gpu_result = {
        let mut raw_outputs = unsafe { presize_and_view_base::<E>(columns, lde_size) };
        math_cuda::lde::coset_lde_batch_base_into(
            &slices,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
        )
    };
    if gpu_result.is_err() {
        // Restore columns to trace length for the CPU fallback. `math_cuda`
        // only writes outputs at the very end (post-synchronize host copy);
        // on any Err the caller's `columns[0..n]` is untouched trace data.
        restore_columns_on_err(columns, n);
        return false;
    }
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
    domain: &Domain<F>,
) -> Option<(Vec<FieldElement<E>>, Vec<FieldElement<E>>)>
where
    F: math::field::traits::IsFFTField + IsField + 'static,
    E: IsField + 'static,
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
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    GPU_EXTEND_HALVES_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Weights are built from `g = domain.coset_offset` directly: the
    // CPU caller previously passed `g²` redundantly. See the
    // `g^(-k) / N` weight loop below.

    // Flatten ext3 slices to raw 3*n u64 buffers.
    let to_u64 = |col: &[FieldElement<E>]| -> Vec<u64> {
        let len = col.len() * 3;
        let ptr = col.as_ptr() as *const u64;
        unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
    };
    let h0_raw = to_u64(h0);
    let h1_raw = to_u64(h1);

    // weights[k] = g^(-k) / N as a u64.
    let inv_n = FieldElement::<F>::from(n as u64).inv().expect("N nonzero");
    let g = &domain.coset_offset;
    let g_inv = g.inv().expect("g nonzero");
    let mut weights_u64 = Vec::with_capacity(n);
    let mut w = inv_n.clone();
    for _ in 0..n {
        // F == GoldilocksField by TypeId check above, so value is u64.
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
        let ext3_len = lde_size
            .checked_mul(3)
            .expect("ext3 output length overflow");
        let out0_slice = unsafe { core::slice::from_raw_parts_mut(out0_ptr, ext3_len) };
        let out1_slice = unsafe { core::slice::from_raw_parts_mut(out1_ptr, ext3_len) };
        let mut outputs: [&mut [u64]; 2] = [out0_slice, out1_slice];
        if math_cuda::lde::coset_lde_batch_ext3_into(&inputs, n, blowup, &weights_u64, &mut outputs)
            .is_err()
        {
            return None;
        }
    }

    Some((lde_h0, lde_h1))
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
    F: IsField + 'static,
    E: IsField + 'static,
{
    let (n, lde_size) = match check_base_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty => return Some(Vec::new()),
        LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_base::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    // Allocate the leaf-hash buffer directly as `Vec<[u8; 32]>` to skip a
    // re-chunk pass; fresh pages fault on first write but only once each.
    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(lde_size);
    // SAFETY: every byte will be overwritten by the GPU D2H below.
    unsafe { leaves.set_len(lde_size) };
    let leaf_byte_len = lde_size.checked_mul(32).expect("leaf byte length overflow");

    GPU_LDE_CALLS.fetch_add(num_columns as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let gpu_result = {
        let mut raw_outputs = unsafe { presize_and_view_base::<E>(columns, lde_size) };
        let hashed_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(leaves.as_mut_ptr() as *mut u8, leaf_byte_len)
        };
        math_cuda::lde::coset_lde_batch_base_into_with_leaf_hash(
            &slices,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
            hashed_bytes,
        )
    };
    if gpu_result.is_err() {
        restore_columns_on_err(columns, n);
        return None;
    }

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
    F: IsField + 'static,
    E: IsField + 'static,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let (n, lde_size) = match check_base_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty | LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();
    let (mut nodes, total_nodes) = alloc_merkle_nodes(lde_size)?;
    let node_byte_len = total_nodes
        .checked_mul(32)
        .expect("node byte length overflow");

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_base::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    GPU_LDE_CALLS.fetch_add(num_columns as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let gpu_result = {
        let mut raw_outputs = unsafe { presize_and_view_base::<E>(columns, lde_size) };
        let nodes_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, node_byte_len)
        };
        math_cuda::lde::coset_lde_batch_base_into_with_merkle_tree(
            &slices,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
            nodes_bytes,
        )
    };
    if gpu_result.is_err() {
        restore_columns_on_err(columns, n);
        return None;
    }

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
    F: IsField + 'static,
    E: IsField + 'static,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let (n, lde_size) = match check_base_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty | LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();
    let (mut nodes, total_nodes) = alloc_merkle_nodes(lde_size)?;
    let node_byte_len = total_nodes
        .checked_mul(32)
        .expect("node byte length overflow");

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_base::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    GPU_LDE_CALLS.fetch_add(num_columns as u64, std::sync::atomic::Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let handle_result = {
        let mut raw_outputs = unsafe { presize_and_view_base::<E>(columns, lde_size) };
        let nodes_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, node_byte_len)
        };
        math_cuda::lde::coset_lde_batch_base_into_with_merkle_tree_keep(
            &slices,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
            nodes_bytes,
        )
    };
    let handle = match handle_result {
        Ok(h) => h,
        Err(_) => {
            restore_columns_on_err(columns, n);
            return None;
        }
    };

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
    F: IsField + 'static,
    E: IsField + 'static,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let (n, lde_size) = match check_ext3_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty | LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();
    let (mut nodes, total_nodes) = alloc_merkle_nodes(lde_size)?;
    let node_byte_len = total_nodes
        .checked_mul(32)
        .expect("node byte length overflow");

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_ext3::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    GPU_LDE_CALLS.fetch_add(
        (num_columns * 3) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let gpu_result = {
        let mut raw_outputs = unsafe { presize_and_view_ext3::<E>(columns, lde_size) };
        let nodes_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, node_byte_len)
        };
        math_cuda::lde::coset_lde_batch_ext3_into_with_merkle_tree(
            &slices,
            n,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
            nodes_bytes,
        )
    };
    if gpu_result.is_err() {
        restore_columns_on_err(columns, n);
        return None;
    }

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
    F: IsField + 'static,
    E: IsField + 'static,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let (n, lde_size) = match check_ext3_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty | LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();
    let (mut nodes, total_nodes) = alloc_merkle_nodes(lde_size)?;
    let node_byte_len = total_nodes
        .checked_mul(32)
        .expect("node byte length overflow");

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_ext3::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    GPU_LDE_CALLS.fetch_add(
        (num_columns * 3) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let handle_result = {
        let mut raw_outputs = unsafe { presize_and_view_ext3::<E>(columns, lde_size) };
        let nodes_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, node_byte_len)
        };
        math_cuda::lde::coset_lde_batch_ext3_into_with_merkle_tree_keep(
            &slices,
            n,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
            nodes_bytes,
        )
    };
    let handle = match handle_result {
        Ok(h) => h,
        Err(_) => {
            restore_columns_on_err(columns, n);
            return None;
        }
    };

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
    F: IsField + 'static,
    E: IsField + 'static,
{
    let (n, lde_size) = match check_ext3_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty => return Some(Vec::new()),
        LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_ext3::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(lde_size);
    // SAFETY: every byte will be overwritten by the GPU D2H below.
    unsafe { leaves.set_len(lde_size) };
    let leaf_byte_len = lde_size.checked_mul(32).expect("leaf byte length overflow");

    GPU_LDE_CALLS.fetch_add(
        (num_columns * 3) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    GPU_LEAF_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let gpu_result = {
        let mut raw_outputs = unsafe { presize_and_view_ext3::<E>(columns, lde_size) };
        let hashed_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(leaves.as_mut_ptr() as *mut u8, leaf_byte_len)
        };
        math_cuda::lde::coset_lde_batch_ext3_into_with_leaf_hash(
            &slices,
            n,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
            hashed_bytes,
        )
    };
    if gpu_result.is_err() {
        restore_columns_on_err(columns, n);
        return None;
    }

    Some(leaves)
}

/// Ext3 specialisation of [`try_expand_columns_batched`]. `E` is known to be
/// `Degree3GoldilocksExtensionField` by TypeId match at the caller.
fn try_expand_columns_batched_ext3<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> bool
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    let (n, lde_size) = match check_ext3_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty => return true,
        LayoutDispatch::Skip => return false,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_ext3::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    // Account each ext3 column as 3 logical GPU LDE "calls" (base-field
    // components) so the counter matches the base-field batched path.
    GPU_LDE_CALLS.fetch_add(
        (num_columns * 3) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let gpu_result = {
        let mut raw_outputs = unsafe { presize_and_view_ext3::<E>(columns, lde_size) };
        math_cuda::lde::coset_lde_batch_ext3_into(
            &slices,
            n,
            blowup_factor,
            &weights_u64,
            &mut raw_outputs,
        )
    };
    if gpu_result.is_err() {
        restore_columns_on_err(columns, n);
        return false;
    }
    true
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
    let leaves_byte_len = leaves_len
        .checked_mul(32)
        .expect("leaf byte length overflow");
    let total_nodes = 2usize
        .checked_mul(leaves_len)
        .and_then(|v| v.checked_sub(1))
        .expect("merkle node count overflow");
    let node_byte_len = total_nodes
        .checked_mul(32)
        .expect("node byte length overflow");

    GPU_MERKLE_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Flatten host-side leaves into a contiguous byte buffer for the GPU
    // kernel. SAFETY: `[u8; 32]` is POD and the slice is contiguous.
    let leaves_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(hashed_leaves.as_ptr() as *const u8, leaves_byte_len)
    };
    let nodes_bytes = match math_cuda::merkle::build_merkle_tree_on_device(leaves_bytes) {
        Ok(b) => b,
        Err(_) => return None,
    };

    debug_assert_eq!(nodes_bytes.len(), node_byte_len);

    // Re-chunk the flat byte buffer into `Vec<[u8; 32]>`. Alignment is
    // identical (`[u8; 32]` has align 1), but `Vec<u8>` and `Vec<[u8; 32]>`
    // track different element counts, so a fresh allocation + per-row copy
    // is the simplest correct conversion.
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
