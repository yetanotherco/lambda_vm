//! GPU dispatch layer for the per-column coset LDE.
//!
//! Handles only Goldilocks base-field columns above a size threshold. Falls
//! back to CPU for extension-field columns and small columns where kernel
//! launch overhead dominates. Produces the same natural-order, non-canonical
//! LDE evaluations as the CPU path.

use core::mem::transmute_copy;
use std::any::TypeId;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use math_cuda::{CudaSlice, CudaStream};

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::FriLayerMerkleTreeBackend;
use crate::domain::Domain;
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_functions::compute_coset_twiddles_inv;
use crate::trace::LDETraceTable;

/// Break-even LDE size. For LDE sizes smaller than this, the CPU
/// `coset_lde_full_expand` completes in a few hundred microseconds and the
/// GPU's tens of kernel launches plus H2D/D2H round-trip is a net loss. The
/// check is on **lde size**, not trace length, because that's what
/// determines the FFT workload.
///
/// 2^19 is a conservative default calibrated against a 46-core machine where
/// rayon-parallel CPU LDE is already fast. Override via env var for tuning
/// on smaller machines, see `crypto/math-cuda/tests/bench_quick.rs`.
const DEFAULT_GPU_LDE_THRESHOLD: usize = 1 << 19;

fn gpu_lde_threshold() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_LDE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_LDE_THRESHOLD)
    })
}

/// Incremented by the `try_expand_*` functions per base-field column handed to
/// the GPU dispatch (an ext3 column counts as 3, one per base component),
/// before the GPU call. A failed call returns without decrementing it, so it
/// counts attempts, not confirmed successes. Used by benchmarks to confirm the
/// GPU path fired.
static GPU_LDE_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn gpu_lde_calls() -> u64 {
    GPU_LDE_CALLS.load(Ordering::Relaxed)
}

/// Reset all GPU call counters at once. Useful between bench warm-up and
/// profiled passes so the numbers reported aren't doubled by the warm-up.
pub fn reset_all_gpu_call_counters() {
    GPU_LDE_CALLS.store(0, Ordering::Relaxed);
    GPU_EXTEND_HALVES_CALLS.store(0, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.store(0, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.store(0, Ordering::Relaxed);
    GPU_PARTS_LDE_CALLS.store(0, Ordering::Relaxed);
    GPU_BARY_CALLS.store(0, Ordering::Relaxed);
    GPU_COMP_POLY_TREE_CALLS.store(0, Ordering::Relaxed);
    GPU_DEEP_CALLS.store(0, Ordering::Relaxed);
    GPU_FRI_CALLS.store(0, Ordering::Relaxed);
    GPU_BATCH_INVERT_CALLS.store(0, Ordering::Relaxed);
    GPU_PAGE_TRACE_CALLS.store(0, Ordering::Relaxed);
    GPU_DECODE_TRACE_CALLS.store(0, Ordering::Relaxed);
    GPU_BITWISE_TRACE_CALLS.store(0, Ordering::Relaxed);
    GPU_LOAD_TRACE_CALLS.store(0, Ordering::Relaxed);
}

/// PAGE-table GPU dispatch counter. Incremented once per
/// `generate_page_trace` call that successfully ran on GPU. Used by the
/// trace-expansion integration tests to verify the GPU path fired.
pub(crate) static GPU_PAGE_TRACE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_page_trace_calls() -> u64 {
    GPU_PAGE_TRACE_CALLS.load(Ordering::Relaxed)
}

/// Prover-crate wrapper for the GPU page-trace generator. Hides
/// `math_cuda` from the prover so the prover crate doesn't need to
/// declare it as a direct dependency. Returns `None` on any cudarc /
/// backend failure so the caller's CPU loop runs unchanged.
pub fn try_generate_page_trace_gpu_raw(
    page_size: usize,
    init_values: &[u64],
    final_values: &[u64],
    final_timestamps: &[u64],
    num_cols: usize,
) -> Option<Vec<u64>> {
    let raw = math_cuda::page_trace::generate_page_trace_dev(
        page_size,
        init_values,
        final_values,
        final_timestamps,
        num_cols,
    )
    .ok()?;
    GPU_PAGE_TRACE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(raw)
}

/// DECODE-table GPU dispatch counter. Bumped per successful
/// `generate_decode_trace` GPU call.
pub(crate) static GPU_DECODE_TRACE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_decode_trace_calls() -> u64 {
    GPU_DECODE_TRACE_CALLS.load(Ordering::Relaxed)
}

/// Prover-crate wrapper for the GPU DECODE-trace generator (same shape
/// as `try_generate_page_trace_gpu_raw`).
pub fn try_generate_decode_trace_gpu_raw(
    num_rows: usize,
    pcs: &[u64],
    packed_decodes: &[u64],
    imms: &[u64],
    num_cols: usize,
) -> Option<Vec<u64>> {
    let raw = math_cuda::decode_trace::generate_decode_trace_dev(
        num_rows,
        pcs,
        packed_decodes,
        imms,
        num_cols,
    )
    .ok()?;
    GPU_DECODE_TRACE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(raw)
}

/// BITWISE-table GPU dispatch counter.
pub(crate) static GPU_BITWISE_TRACE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_bitwise_trace_calls() -> u64 {
    GPU_BITWISE_TRACE_CALLS.load(Ordering::Relaxed)
}

/// Prover-crate wrapper for the GPU BITWISE-trace generator. Preprocessed
/// table — kernel produces the full 21-column row-major buffer with
/// multiplicity columns at 0. Caller still runs `update_multiplicities`
/// on the host TraceTable.
pub fn try_generate_bitwise_trace_gpu_raw(
    num_rows: usize,
    num_cols: usize,
) -> Option<Vec<u64>> {
    let raw = math_cuda::bitwise_trace::generate_bitwise_trace_dev(num_rows, num_cols).ok()?;
    GPU_BITWISE_TRACE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(raw)
}

/// LOAD-table GPU dispatch counter.
pub(crate) static GPU_LOAD_TRACE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_load_trace_calls() -> u64 {
    GPU_LOAD_TRACE_CALLS.load(Ordering::Relaxed)
}

/// Prover-crate wrapper for the GPU LOAD-trace generator. `flags` is bit-
/// packed per row: bit0=READ2 bit1=READ4 bit2=READ8 bit3=SIGNED
/// bit4=SIGN_BIT bit5=MU. `res_bytes` is `8 * num_rows` interleaved.
pub fn try_generate_load_trace_gpu_raw(
    num_rows: usize,
    base_addresses: &[u64],
    timestamps: &[u64],
    flags: &[u64],
    res_bytes: &[u64],
    num_cols: usize,
) -> Option<Vec<u64>> {
    let raw = math_cuda::load_trace::generate_load_trace_dev(
        num_rows,
        base_addresses,
        timestamps,
        flags,
        res_bytes,
        num_cols,
    )
    .ok()?;
    GPU_LOAD_TRACE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(raw)
}

pub(crate) static GPU_EXTEND_HALVES_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_extend_halves_calls() -> u64 {
    GPU_EXTEND_HALVES_CALLS.load(Ordering::Relaxed)
}

// ============================================================================
// Shared dispatch helpers
// ============================================================================
//
// Common prologue for the try_expand_* variants: empty-check, threshold,
// TypeId checks, equal-length check, column-to-u64 cast.

/// Outcome of validating an input slice against the GPU dispatch preconditions.
enum LayoutDispatch {
    /// Input slice is empty, no work to do.
    Empty,
    /// Preconditions not met: below threshold, wrong element types, or
    /// columns of unequal length.
    Skip,
    /// Preconditions met. `n` is the per-column input length:
    /// `lde_size = n * blowup_factor` (saturating).
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

/// Convert base-field columns to `Vec<Vec<u64>>` for the GPU input slice list.
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

/// Convert ext3 columns to `Vec<Vec<u64>>`, each column reinterpreted as a flat
/// `u64` slice with the three coordinates of every element kept contiguous
/// (`[a0, a1, a2, b0, b1, b2, ...]`), for the GPU input slice list.
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
            unsafe { from_raw_parts(ptr, len) }.to_vec()
        })
        .collect()
}

/// Convert weights to raw `Vec<u64>`.
///
/// SAFETY: caller must have established `F == GoldilocksField`.
unsafe fn weights_to_u64<F: IsField>(weights: &[FieldElement<F>]) -> Vec<u64> {
    weights
        .iter()
        .map(|w| unsafe { *(w.value() as *const _ as *const u64) })
        .collect()
}

/// Pre-size each column to `lde_size` and view it as a `&mut [u64]` of length
/// `lde_size` (base-field, single-u64 layout).
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
        // SAFETY: assert above guarantees capacity, the GPU path overwrites
        // every slot before any reader sees the new length.
        unsafe { col.set_len(lde_size) };
    }
    columns
        .iter_mut()
        .map(|col| {
            let ptr = col.as_mut_ptr() as *mut u64;
            let len = col.len();
            // SAFETY: single-u64 layout, caller still owns the backing alloc.
            unsafe { from_raw_parts_mut(ptr, len) }
        })
        .collect()
}

/// Same as [`presize_and_view_base`] but for ext3 columns: each view is
/// `3 * lde_size` u64s, the three coordinates of every element contiguous.
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
            unsafe { from_raw_parts_mut(ptr, len) }
        })
        .collect()
}

/// Truncate each column back to `n` (trace size) after a GPU error so the
/// CPU fallback (which reads `buffer.len()` as the trace size) runs cleanly.
/// Safe because `math_cuda` writes outputs only at the final host copy, post-
/// synchronize; any `Err` returns before that copy, leaving `columns[0..n]` untouched.
fn restore_columns_on_err<E: IsField>(columns: &mut [Vec<FieldElement<E>>], n: usize) {
    for col in columns.iter_mut() {
        col.truncate(n);
    }
}

/// Allocate the `[u8; 32]` Merkle node buffer for a tree of `lde_size` leaves
/// and return the node `Vec` (length-initialised, contents undefined) together
/// with its node count `total_nodes` (`2 * lde_size - 1`). Returns `None` if
/// the layout would be invalid (`lde_size < 2` or `total_nodes * 32` overflows
/// `usize`). The caller builds the `&mut [u8]` byte view of length
/// `total_nodes * 32` and must overwrite every byte via the GPU D2H.
fn alloc_merkle_nodes(lde_size: usize) -> Option<(Vec<[u8; 32]>, usize)> {
    if lde_size < 2 {
        return None;
    }
    let total_nodes = 2usize.saturating_mul(lde_size).checked_sub(1)?;
    let _byte_len = total_nodes.checked_mul(32)?;
    let mut nodes: Vec<[u8; 32]> = Vec::with_capacity(total_nodes);
    // SAFETY: every byte will be overwritten via the GPU D2H before the
    // contents are read. The caller computes the byte-length view from the
    // returned `nodes` Vec using `total_nodes.checked_mul(32)`.
    #[allow(clippy::uninit_vec)]
    unsafe {
        nodes.set_len(total_nodes)
    };
    Some((nodes, total_nodes))
}

/// Try to GPU-batch all columns in one pass.
///
/// Engaged for Goldilocks-base and ext3 tables whose LDE size is above the
/// threshold (ext3 is routed to [`try_expand_columns_batched_ext3`]). The
/// prover's `expand_columns_to_lde` hands us every column of one table at
/// once. Those columns all share twiddles and coset weights so they can be
/// processed in a single batched pipeline on one stream.
///
/// Returns `Some(())` if the batch was handled on GPU and `columns` now holds
/// the LDE evaluations, or if there were no columns to expand. Returns `None`
/// to let the caller run the per-column CPU fallback.
pub(crate) fn try_expand_columns_batched<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<()>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    // Ext3 columns go through the ext3 specialization.
    if TypeId::of::<E>() == TypeId::of::<Degree3GoldilocksExtensionField>() {
        return try_expand_columns_batched_ext3::<F, E>(columns, blowup_factor, weights);
    }

    let (n, lde_size) = match check_base_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty => return Some(()), // nothing to do — same as CPU path
        LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();

    // SAFETY: the `Run` arm of `check_base_layout::<F, E>` (matched above)
    // guarantees `E == GoldilocksField` and `F == GoldilocksField`.
    let raw_columns = unsafe { columns_to_u64_base::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();
    GPU_LDE_CALLS.fetch_add(num_columns as u64, Ordering::Relaxed);
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
        return None;
    }
    Some(())
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
#[allow(clippy::type_complexity)]
pub(crate) fn try_extend_two_halves_gpu<F, E>(
    h0: &[FieldElement<E>],
    h1: &[FieldElement<E>],
    domain: &Domain<F>,
) -> Option<(Vec<FieldElement<E>>, Vec<FieldElement<E>>)>
where
    F: IsFFTField + IsField + 'static,
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
    GPU_EXTEND_HALVES_CALLS.fetch_add(1, Ordering::Relaxed);
    // Weights are built from `g = domain.coset_offset` directly: the
    // CPU caller previously passed `g²` redundantly. See the
    // `g^(-k) / N` weight loop below.

    // Flatten ext3 slices to raw 3*n u64 buffers.
    let to_u64 = |col: &[FieldElement<E>]| -> Vec<u64> {
        let len = col.len() * 3;
        let ptr = col.as_ptr() as *const u64;
        unsafe { from_raw_parts(ptr, len) }.to_vec()
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
        w *= &g_inv;
    }

    // Pre-allocate outputs.
    let mut lde_h0 = vec![FieldElement::<E>::zero(); lde_size];
    let mut lde_h1 = vec![FieldElement::<E>::zero(); lde_size];

    // Two ext3 columns (h0 + h1), each composed of 3 base-field components.
    const NUM_COLS: usize = 2;
    GPU_LDE_CALLS.fetch_add((NUM_COLS * 3) as u64, Ordering::Relaxed);
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
        let out0_slice = unsafe { from_raw_parts_mut(out0_ptr, ext3_len) };
        let out1_slice = unsafe { from_raw_parts_mut(out1_ptr, ext3_len) };
        let mut outputs: [&mut [u64]; 2] = [out0_slice, out1_slice];
        if math_cuda::lde::coset_lde_batch_ext3_into(&inputs, n, blowup, &weights_u64, &mut outputs)
            .is_err()
        {
            return None;
        }
    }

    Some((lde_h0, lde_h1))
}

pub(crate) static GPU_LEAF_HASH_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_leaf_hash_calls() -> u64 {
    GPU_LEAF_HASH_CALLS.load(Ordering::Relaxed)
}

/// Fused base-field path: LDE + Keccak-256 leaf hash + Merkle tree build,
/// all on device, with the LDE buffer retained for R2–R4 GPU reuse. On
/// success: `columns[c]` is resized to `lde_size` with the LDE output, and
/// the returned `(tree, GpuLdeBase)` pair is the host-side tree plus a
/// device-resident handle to the LDE buffer.
pub(crate) fn try_expand_leaf_and_tree_batched_keep<F, E, B>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<(MerkleTree<B>, math_cuda::lde::GpuLdeBase)>
where
    F: IsField + 'static,
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
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

    GPU_LDE_CALLS.fetch_add(num_columns as u64, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, Ordering::Relaxed);

    let handle_result = {
        let mut raw_outputs = unsafe { presize_and_view_base::<E>(columns, lde_size) };
        let nodes_bytes: &mut [u8] =
            unsafe { from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, node_byte_len) };
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

    let tree = MerkleTree::<B>::from_precomputed_nodes(nodes)?;
    Some((tree, handle))
}

/// Fused ext3 path: LDE + Keccak-256 leaf hash + Merkle tree build over
/// ext3 columns via the three-slab decomposition, with the ext3 LDE device
/// buffer (de-interleaved 3-slab layout) retained for downstream GPU rounds.
/// `B::Node = [u8; 32]` by construction for `BatchKeccak256Backend<Ext3>`.
pub(crate) fn try_expand_leaf_and_tree_batched_ext3_keep<F, E, B>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<(MerkleTree<B>, math_cuda::lde::GpuLdeExt3)>
where
    F: IsField + 'static,
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
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

    GPU_LDE_CALLS.fetch_add((num_columns * 3) as u64, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, Ordering::Relaxed);

    let handle_result = {
        let mut raw_outputs = unsafe { presize_and_view_ext3::<E>(columns, lde_size) };
        let nodes_bytes: &mut [u8] =
            unsafe { from_raw_parts_mut(nodes.as_mut_ptr() as *mut u8, node_byte_len) };
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

    let tree = MerkleTree::<B>::from_precomputed_nodes(nodes)?;
    Some((tree, handle))
}

/// Ext3 specialisation of [`try_expand_columns_batched`]. `E` is known to be
/// `Degree3GoldilocksExtensionField` by TypeId match at the caller.
///
/// The LDE runs over the 3 base-field components of each ext3 column. The
/// transform uses only base-field twiddles and coset weights, which act
/// componentwise on ext3, so the per-component result equals the ext3 LDE the
/// CPU path computes.
fn try_expand_columns_batched_ext3<F, E>(
    columns: &mut [Vec<FieldElement<E>>],
    blowup_factor: usize,
    weights: &[FieldElement<F>],
) -> Option<()>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    let (n, lde_size) = match check_ext3_layout::<F, E>(columns, blowup_factor) {
        LayoutDispatch::Empty => return Some(()),
        LayoutDispatch::Skip => return None,
        LayoutDispatch::Run { n, lde_size } => (n, lde_size),
    };
    let num_columns = columns.len();

    // SAFETY: layout-checked above.
    let raw_columns = unsafe { columns_to_u64_ext3::<E>(columns) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };
    let slices: Vec<&[u64]> = raw_columns.iter().map(|c| c.as_slice()).collect();

    // Account each ext3 column as 3 logical GPU LDE "calls" (base-field
    // components) so the counter matches the base-field batched path.
    GPU_LDE_CALLS.fetch_add((num_columns * 3) as u64, Ordering::Relaxed);
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
        return None;
    }
    Some(())
}

static GPU_MERKLE_TREE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_merkle_tree_calls() -> u64 {
    GPU_MERKLE_TREE_CALLS.load(Ordering::Relaxed)
}

// ============================================================================
// PR-3: R2 composition-parts LDE + Merkle commit + R3 OOD barycentric
// ============================================================================

/// R2 dispatch counter: incremented once per
/// [`try_evaluate_parts_on_lde_gpu_keep`] call that actually routed to the GPU.
pub(crate) static GPU_PARTS_LDE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_parts_lde_calls() -> u64 {
    GPU_PARTS_LDE_CALLS.load(Ordering::Relaxed)
}

/// R3 dispatch counter: incremented once per `try_barycentric_*_on_handle`
/// call that actually routed to the GPU.
pub(crate) static GPU_BARY_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_bary_calls() -> u64 {
    GPU_BARY_CALLS.load(Ordering::Relaxed)
}

/// R2 comp-poly tree counter: incremented once per
/// [`try_build_comp_poly_tree_gpu`] call that actually routed to the GPU.
/// Distinct from `GPU_MERKLE_TREE_CALLS` (R1 main/aux tree builds) so the
/// two dispatch sites can be diagnosed independently.
pub(crate) static GPU_COMP_POLY_TREE_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_comp_poly_tree_calls() -> u64 {
    GPU_COMP_POLY_TREE_CALLS.load(Ordering::Relaxed)
}

/// Trace-size threshold for the R3 OOD barycentric GPU path. Below this the
/// rayon CPU path is competitive and PCIe round-trip overhead would dominate.
/// Override via `LAMBDA_VM_GPU_BARY_THRESHOLD`.
const DEFAULT_GPU_BARY_THRESHOLD: usize = 1 << 14;

fn gpu_bary_threshold() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_BARY_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_BARY_THRESHOLD)
    })
}

/// One ext3 scalar `(n_inv * g_n_inv) * (z^N - g^N)`. Returned as `[u64;3]`.
///
/// The GPU kernels compute only the unscaled barycentric sum per column.
/// Applying this scalar on the host is one ext3 multiply per column, cheap
/// next to the trace-size reduction.
fn ood_ext3_scalar<F, E>(
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
) -> [u64; 3]
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    // (z^N - g^N) in E, via sub_subfield (E - F -> E).
    let vanishing = z_pow_n.sub_subfield(coset_offset_pow_n);
    let base_scalar = n_inv * g_n_inv; // F * F -> F
    let scalar_ext3: FieldElement<E> = &base_scalar * &vanishing; // F * E -> E
    // SAFETY: TypeId-checked at the caller (`try_barycentric_*_on_handle`).
    // E == Degree3GoldilocksExtensionField, whose backing is
    // `[FieldElement<Gl>; 3]`, layout-equivalent to `[u64; 3]`.
    let ptr = &scalar_ext3 as *const FieldElement<E> as *const u64;
    unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] }
}

/// Multiply each raw GPU ext3 sum by the host-computed ext3 scalar. `sums_raw`
/// is `3 * num_cols` u64s (interleaved). Returns the final OOD evaluations as
/// `Vec<FieldElement<E>>` of length `num_cols`.
fn apply_ext3_scalar<E>(sums_raw: &[u64], scalar: [u64; 3], num_cols: usize) -> Vec<FieldElement<E>>
where
    E: IsField + 'static,
{
    type Gl = GoldilocksField;
    type Ext3 = Degree3GoldilocksExtensionField;

    assert_eq!(sums_raw.len(), 3 * num_cols);
    // Avoids the `E != Ext3` path reaching the unsafe `transmute_copy` below
    // that is UB in that case. Cost is one TypeId comparison per call.
    assert_eq!(
        TypeId::of::<E>(),
        TypeId::of::<Ext3>(),
        "apply_ext3_scalar: E must be Ext3"
    );

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
        let final_ext3 = s * scalar_e;
        // SAFETY: TypeId-checked at the caller. E == Ext3, identical layout.
        let final_e: FieldElement<E> =
            unsafe { transmute_copy::<FieldElement<Ext3>, FieldElement<E>>(&final_ext3) };
        out.push(final_e);
    }
    out
}

/// R2 GPU dispatch: build the composition-polynomial Merkle tree from the
/// host-side ext3 LDE eval Vecs produced by
/// [`try_evaluate_parts_on_lde_gpu_keep`] (or the CPU path). Uses the same
/// row-pair leaf pattern as the CPU
/// `commit_composition_polynomial`: each leaf hashes 2 consecutive
/// bit-reversed rows.
///
/// Returns `None` to fall through to the CPU path when the type or size
/// conditions don't hold; returns `None` on a math-cuda `Err` so the caller
/// recomputes on CPU.
pub(crate) fn try_build_comp_poly_tree_gpu<E, B>(
    lde_parts: &[Vec<FieldElement<E>>],
) -> Option<MerkleTree<B>>
where
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
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
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    // All parts must have the same LDE length.
    if lde_parts.iter().any(|p| p.len() != lde_size) {
        return None;
    }

    // SAFETY: E == Ext3 per TypeId check. FieldElement<Ext3> backing is
    // `[FieldElement<Gl>; 3]`, layout-equivalent to `[u64; 3]`.
    let raw_parts: Vec<&[u64]> = lde_parts
        .iter()
        .map(|p| {
            let byte_len = p.len().checked_mul(3).expect("ext3 part byte len overflow");
            unsafe { from_raw_parts(p.as_ptr() as *const u64, byte_len) }
        })
        .collect();

    let nodes_bytes = match math_cuda::merkle::build_comp_poly_tree_from_evals_ext3(&raw_parts) {
        Ok(v) => v,
        Err(_) => return None,
    };

    // lde_size is an even power of two >= 2, so 2*num_leaves == lde_size and
    // tight_total_nodes = lde_size - 1 >= 1. No overflow or underflow possible.
    let tight_total_nodes = lde_size - 1;
    let expected_byte_len = tight_total_nodes
        .checked_mul(32)
        .expect("comp-poly node byte length overflow");
    debug_assert_eq!(nodes_bytes.len(), expected_byte_len);

    let nodes: Vec<[u8; 32]> = nodes_bytes
        .chunks_exact(32)
        .map(|c| {
            c.try_into()
                .expect("chunks_exact(32) yields exactly 32 bytes")
        })
        .collect();
    GPU_COMP_POLY_TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    // Falls back to CPU on `None`, matching the R1 paths (lines 496, 557).
    MerkleTree::<B>::from_precomputed_nodes(nodes)
}

/// R3 GPU dispatch: batched strided barycentric OOD evaluation over the main
/// (base-field) LDE columns kept on device from R1. Operates on the
/// device-resident LDE in place; only the coset points and inv_denoms are
/// copied to the device, not the columns. Returns the OOD evaluations
/// as `Vec<FieldElement<E>>` of length `num_main_cols` (already scaled by
/// `vanishing * n_inv * g_n_inv`), or `None` if the GPU handle is absent,
/// types don't match, the trace-size domain is below threshold, or the
/// math-cuda call returns `Err`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_base_on_handle<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    row_stride: usize,
    coset_points: &[FieldElement<F>],
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
    inv_denoms_host: &[FieldElement<E>],
    r3_ctx: Option<(&R3DevContext, usize)>,
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
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
    if main.lde_size != n.checked_mul(row_stride)? {
        return None;
    }
    // Host inv_denoms length only matters on the host path.
    if r3_ctx.is_none() && inv_denoms_host.len() != n {
        return None;
    }

    // SAFETY: F == Goldilocks per TypeId check; FieldElement<Gl> is
    // #[repr(transparent)] over u64.
    let points_raw: &[u64] = unsafe { from_raw_parts(coset_points.as_ptr() as *const u64, n) };

    let sums_raw = match r3_ctx {
        Some((ctx, inv_offset_u64)) => {
            match math_cuda::barycentric::barycentric_base_on_device_with_dev_inv_denoms(
                &ctx.stream,
                main,
                row_stride,
                &ctx.coset_points,
                &ctx.inv_denoms,
                inv_offset_u64,
                n,
            ) {
                Ok(v) => v,
                Err(_) => return None,
            }
        }
        None => {
            // SAFETY: E == Ext3 per TypeId check; FieldElement<Ext3> backing is `[u64; 3]`.
            let inv_denoms_len = n.checked_mul(3).expect("inv_denoms u64 len overflow");
            let inv_denoms_raw: &[u64] =
                unsafe { from_raw_parts(inv_denoms_host.as_ptr() as *const u64, inv_denoms_len) };
            match math_cuda::barycentric::barycentric_base_on_device(
                main,
                row_stride,
                points_raw,
                inv_denoms_raw,
                n,
            ) {
                Ok(v) => v,
                Err(_) => return None,
            }
        }
    };
    GPU_BARY_CALLS.fetch_add(1, Ordering::Relaxed);

    let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
    Some(apply_ext3_scalar::<E>(&sums_raw, scalar, num_cols))
}

/// Ext3 counterpart of [`try_barycentric_base_on_handle`] for the aux LDE.
/// Reads `lde_trace.gpu_aux()` (the de-interleaved 3-slab device buffer).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_ext3_on_handle<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    row_stride: usize,
    coset_points: &[FieldElement<F>],
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pow_n: &FieldElement<E>,
    inv_denoms_host: &[FieldElement<E>],
    r3_ctx: Option<(&R3DevContext, usize)>,
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
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
    if aux.lde_size != n.checked_mul(row_stride)? {
        return None;
    }
    if r3_ctx.is_none() && inv_denoms_host.len() != n {
        return None;
    }

    let points_raw: &[u64] = unsafe { from_raw_parts(coset_points.as_ptr() as *const u64, n) };

    let sums_raw = match r3_ctx {
        Some((ctx, inv_offset_u64)) => {
            match math_cuda::barycentric::barycentric_ext3_on_device_with_dev_inv_denoms(
                &ctx.stream,
                aux,
                row_stride,
                &ctx.coset_points,
                &ctx.inv_denoms,
                inv_offset_u64,
                n,
            ) {
                Ok(v) => v,
                Err(_) => return None,
            }
        }
        None => {
            let inv_denoms_len = n.checked_mul(3).expect("inv_denoms u64 len overflow");
            let inv_denoms_raw: &[u64] =
                unsafe { from_raw_parts(inv_denoms_host.as_ptr() as *const u64, inv_denoms_len) };
            match math_cuda::barycentric::barycentric_ext3_on_device(
                aux,
                row_stride,
                points_raw,
                inv_denoms_raw,
                n,
            ) {
                Ok(v) => v,
                Err(_) => return None,
            }
        }
    };
    GPU_BARY_CALLS.fetch_add(1, Ordering::Relaxed);

    let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
    Some(apply_ext3_scalar::<E>(&sums_raw, scalar, num_cols))
}

// ============================================================================
// R2 keep-handle variant, R4 DEEP composition, FRI commit dispatches
// ============================================================================

/// R4 DEEP-composition dispatch counter.
pub(crate) static GPU_DEEP_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_deep_calls() -> u64 {
    GPU_DEEP_CALLS.load(Ordering::Relaxed)
}

/// FRI commit-phase dispatch counter (one per `try_fri_commit_gpu` call,
/// not per layer).
pub(crate) static GPU_FRI_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_fri_calls() -> u64 {
    GPU_FRI_CALLS.load(Ordering::Relaxed)
}

/// Batch-invert dispatch counter (one per
/// [`try_compute_and_invert_inv_denoms_dev`] call that actually built a
/// device handle). Fires at most twice per prove per table: once for R3
/// OOD's `num_eval_points * trace_size` denominators and once for R4
/// DEEP's `(1 + num_eval_points) * lde_size` denominators.
pub(crate) static GPU_BATCH_INVERT_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_batch_invert_calls() -> u64 {
    GPU_BATCH_INVERT_CALLS.load(Ordering::Relaxed)
}

/// Test-only: schedule the Nth upcoming FRI fold call (1 = first, 2 =
/// second, ...) to return Err, exercising the snapshot-restore path in
/// [`try_fri_commit_gpu`]. Pass -1 to disable. Production default is -1.
/// Only available with the `test-cuda-faults` feature.
#[cfg(feature = "test-cuda-faults")]
pub fn schedule_fri_fold_fault(n_calls_until_err: i64) {
    math_cuda::fri::FAULT_FOLDS_REMAINING_UNTIL_ERR.store(n_calls_until_err, Ordering::Relaxed);
}

/// Test-only: schedule the Nth upcoming `compute_and_invert_denoms_ext3_dev`
/// call to return Err, exercising the CPU-fallback path in
/// [`try_compute_and_invert_inv_denoms_dev`]. Pass -1 to disable.
#[cfg(feature = "test-cuda-faults")]
pub fn schedule_inverse_fault(n_calls_until_err: i64) {
    math_cuda::inverse::FAULT_INVERSE_REMAINING_UNTIL_ERR
        .store(n_calls_until_err, Ordering::Relaxed);
}

/// Test-only: schedule the Nth upcoming `generate_page_trace_dev` call to
/// return Err. Pass -1 to disable.
#[cfg(feature = "test-cuda-faults")]
pub fn schedule_page_trace_fault(n_calls_until_err: i64) {
    math_cuda::page_trace::FAULT_PAGE_TRACE_REMAINING_UNTIL_ERR
        .store(n_calls_until_err, Ordering::Relaxed);
}

/// Test-only: schedule the Nth upcoming `generate_decode_trace_dev` call to
/// return Err. Pass -1 to disable.
#[cfg(feature = "test-cuda-faults")]
pub fn schedule_decode_trace_fault(n_calls_until_err: i64) {
    math_cuda::decode_trace::FAULT_DECODE_TRACE_REMAINING_UNTIL_ERR
        .store(n_calls_until_err, Ordering::Relaxed);
}

/// Test-only: schedule the Nth upcoming `generate_bitwise_trace_dev` call to Err.
#[cfg(feature = "test-cuda-faults")]
pub fn schedule_bitwise_trace_fault(n_calls_until_err: i64) {
    math_cuda::bitwise_trace::FAULT_BITWISE_TRACE_REMAINING_UNTIL_ERR
        .store(n_calls_until_err, Ordering::Relaxed);
}

/// Test-only: schedule the Nth upcoming `generate_load_trace_dev` call to Err.
#[cfg(feature = "test-cuda-faults")]
pub fn schedule_load_trace_fault(n_calls_until_err: i64) {
    math_cuda::load_trace::FAULT_LOAD_TRACE_REMAINING_UNTIL_ERR
        .store(n_calls_until_err, Ordering::Relaxed);
}

/// R2 GPU dispatch: batched ext3 LDE over `parts_coefs` (composition-poly
/// coefficient parts). Returns both the host LDE eval Vecs (needed for the
/// R2 Merkle commit and R3 OOD path) and a device-resident `GpuLdeExt3`
/// handle to the same de-interleaved buffer, so R4 DEEP can skip the
/// `num_parts * 3 * lde_size * 8` byte H2D.
pub(crate) fn try_evaluate_parts_on_lde_gpu_keep<F, E>(
    parts_coefs: &[&[FieldElement<E>]],
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
) -> Option<(Vec<Vec<FieldElement<E>>>, math_cuda::lde::GpuLdeExt3)>
where
    F: IsFFTField + IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if parts_coefs.is_empty() {
        return None;
    }
    if !domain_size.is_power_of_two() || !blowup_factor.is_power_of_two() {
        return None;
    }
    let lde_size = domain_size.checked_mul(blowup_factor)?;
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    let m = parts_coefs.len();

    let mut weights_u64 = Vec::with_capacity(domain_size);
    let mut w = FieldElement::<F>::one();
    for _ in 0..domain_size {
        // SAFETY: F == Goldilocks per TypeId check.
        let v: u64 = unsafe { *(w.value() as *const _ as *const u64) };
        weights_u64.push(v);
        w *= offset;
    }

    let mut part_bufs: Vec<Vec<u64>> = Vec::with_capacity(m);
    for part in parts_coefs.iter() {
        let mut buf = vec![0u64; 3 * domain_size];
        let len = part.len().min(domain_size);
        // SAFETY: E == Ext3 per TypeId check; backing is `[u64; 3]`.
        let src_ptr = part.as_ptr() as *const u64;
        let src_len = len.checked_mul(3).expect("part src len overflow");
        let src = unsafe { from_raw_parts(src_ptr, src_len) };
        buf[..src_len].copy_from_slice(src);
        part_bufs.push(buf);
    }
    let input_slices: Vec<&[u64]> = part_bufs.iter().map(|v| v.as_slice()).collect();

    let mut outputs: Vec<Vec<FieldElement<E>>> = (0..m)
        .map(|_| vec![FieldElement::<E>::zero(); lde_size])
        .collect();
    let handle_result = {
        let mut out_slices: Vec<&mut [u64]> = outputs
            .iter_mut()
            .map(|o| {
                let ptr = o.as_mut_ptr() as *mut u64;
                let byte_len = lde_size.checked_mul(3).expect("ext3 out len overflow");
                // SAFETY: E == Ext3; Vec of `lde_size` ext3 is `3*lde_size` u64s.
                unsafe { from_raw_parts_mut(ptr, byte_len) }
            })
            .collect();
        math_cuda::lde::evaluate_poly_coset_batch_ext3_into_keep(
            &input_slices,
            domain_size,
            blowup_factor,
            &weights_u64,
            &mut out_slices,
        )
    };
    let handle = match handle_result {
        Ok(h) => h,
        Err(_) => return None,
    };
    GPU_PARTS_LDE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((outputs, handle))
}

/// Reinterpret a slice of ext3 `FieldElement`s as a raw `&[u64]` of length
/// `3 * col.len()`. Caller must have established `E == Ext3` (TypeId check).
///
/// SAFETY: `E == Degree3GoldilocksExtensionField` so each element is
/// `[FieldElement<Gl>; 3]` = `[u64; 3]`.
unsafe fn ext3_slice_to_u64<E: IsField>(col: &[FieldElement<E>]) -> &[u64] {
    let len = col.len().checked_mul(3).expect("ext3 u64 len overflow");
    let ptr = col.as_ptr() as *const u64;
    unsafe { from_raw_parts(ptr, len) }
}

/// Convert ext3 evals (3*n u64s, interleaved) into a freshly allocated
/// `Vec<FieldElement<E>>` of length `n`. Caller must have established
/// `E == Ext3`.
fn u64_to_ext3_vec<E>(raw: &[u64]) -> Vec<FieldElement<E>>
where
    E: IsField + 'static,
{
    type Gl = GoldilocksField;
    type Ext3 = Degree3GoldilocksExtensionField;
    assert_eq!(TypeId::of::<E>(), TypeId::of::<Ext3>());
    assert!(raw.len().is_multiple_of(3));
    let n = raw.len() / 3;
    let mut out: Vec<FieldElement<E>> = Vec::with_capacity(n);
    for i in 0..n {
        let v: FieldElement<Ext3> = FieldElement::<Ext3>::new([
            FieldElement::<Gl>::from_raw(raw[i * 3]),
            FieldElement::<Gl>::from_raw(raw[i * 3 + 1]),
            FieldElement::<Gl>::from_raw(raw[i * 3 + 2]),
        ]);
        // SAFETY: TypeId-checked above. E == Ext3, identical layout.
        out.push(unsafe { transmute_copy::<FieldElement<Ext3>, FieldElement<E>>(&v) });
    }
    out
}

/// R4 GPU dispatch: per-row DEEP composition over the full LDE domain.
/// Reuses the device-resident main + (optional) aux LDE handles from R1
/// and, when supplied, the device-resident composition-parts LDE handle
/// from the R2 `_keep` path.
///
/// Returns the `lde_size` ext3 evaluations of the DEEP polynomial on
/// success, or `None` to let the caller run its existing CPU loop. The
/// caller's `inv_denoms` must be `inv_denoms[0..lde_size]` for the H-term
/// and `inv_denoms[(1+k)*lde_size..(2+k)*lde_size]` for trace term k
/// (matching `compute_deep_composition_poly_evaluations`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_deep_composition_gpu<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    parts_dev: Option<&math_cuda::lde::GpuLdeExt3>,
    parts_host: &[Vec<FieldElement<E>>],
    h_ood: &[FieldElement<E>],
    trace_ood_columns: &[Vec<FieldElement<E>>],
    composition_poly_gammas: &[FieldElement<E>],
    trace_terms_gammas: &[Vec<FieldElement<E>>],
    inv_denoms_host: &[FieldElement<E>],
    inv_denoms_dev: Option<(&CudaSlice<u64>, &Arc<CudaStream>)>,
    num_eval_points: usize,
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let main = lde_trace.gpu_main()?;
    let lde_size = main.lde_size;
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if !lde_size.is_power_of_two() {
        return None;
    }
    let num_main = main.m;
    let aux_handle = lde_trace.gpu_aux();
    let num_aux = aux_handle.map(|a| a.m).unwrap_or(0);
    let num_total_cols = num_main + num_aux;
    let num_parts = composition_poly_gammas.len();
    if h_ood.len() != num_parts {
        return None;
    }
    if trace_ood_columns.len() != num_total_cols
        || trace_ood_columns.iter().any(|c| c.len() != num_eval_points)
    {
        return None;
    }
    if trace_terms_gammas.len() != num_total_cols
        || trace_terms_gammas
            .iter()
            .any(|c| c.len() != num_eval_points)
    {
        return None;
    }
    let expected_inv_denoms = lde_size.checked_mul(1 + num_eval_points)?;
    // The fully-resident `(Some(parts), Some(dev_inv))` arm ignores the
    // host inv_denoms slice; every other arm slices into it. Validate the
    // host length whenever the chosen arm will consume it, even when a
    // dev inv_denoms handle is also present (a (None, Some) combination
    // is reachable when R2's keep path missed but the batch-invert
    // dispatch succeeded; without this guard that path would panic
    // slicing an empty host buffer).
    let arm_needs_host_inv = !(parts_dev.is_some() && inv_denoms_dev.is_some());
    if arm_needs_host_inv && inv_denoms_host.len() != expected_inv_denoms {
        return None;
    }

    // Validate the host parts when we don't have a device handle, since
    // the math-cuda call will assert these.
    if parts_dev.is_none() {
        if parts_host.len() != num_parts {
            return None;
        }
        if parts_host.iter().any(|p| p.len() != lde_size) {
            return None;
        }
    } else if let Some(p) = parts_dev
        && (p.m != num_parts || p.lde_size != lde_size)
    {
        return None;
    }

    // Pack host buffers. SAFETY for ext3 transmutes: E == Ext3 by TypeId check.
    let h_ood_raw: &[u64] = unsafe { ext3_slice_to_u64::<E>(h_ood) };

    // trace_ood: num_total_cols * num_eval_points * 3 (ext3 interleaved,
    // (col * num_eval_points + k) layout).
    let mut trace_ood_raw: Vec<u64> = Vec::with_capacity(num_total_cols * num_eval_points * 3);
    for col in trace_ood_columns {
        let slice = unsafe { ext3_slice_to_u64::<E>(col) };
        trace_ood_raw.extend_from_slice(slice);
    }

    let gammas_h_raw: &[u64] = unsafe { ext3_slice_to_u64::<E>(composition_poly_gammas) };

    let mut gammas_tr_raw: Vec<u64> = Vec::with_capacity(num_total_cols * num_eval_points * 3);
    for col in trace_terms_gammas {
        let slice = unsafe { ext3_slice_to_u64::<E>(col) };
        gammas_tr_raw.extend_from_slice(slice);
    }

    // domain_size == lde_size here: R4 DEEP evaluates at every LDE point
    // (Plonky3-style direct LDE). Calling the kernel with row_stride = 1
    // makes its `row = i * row_stride` index every row.
    let domain_size_kernel = lde_size;
    let row_stride_kernel = 1usize;

    // Three dispatch paths, in priority order:
    //   1. Both parts + inv_denoms on device: the fully-resident path.
    //      Requires the caller's stream so the new inv_denoms_dev producer
    //      and this kernel run on the same queue (no cross-stream race).
    //   2. Parts on device, inv_denoms on host.
    //   3. Both on host (fallback when R2 keep + denom-invert both missed).
    let parts_host_packed: Vec<u64>;
    let result = match (parts_dev, inv_denoms_dev) {
        (Some(parts), Some((inv_dev, stream))) => {
            math_cuda::deep::deep_composition_ext3_with_dev_parts_and_inv_denoms(
                stream,
                main,
                aux_handle,
                parts,
                inv_dev,
                h_ood_raw,
                &trace_ood_raw,
                gammas_h_raw,
                &gammas_tr_raw,
                num_parts,
                num_main,
                num_aux,
                num_eval_points,
                row_stride_kernel,
                domain_size_kernel,
            )
        }
        (Some(parts), None) => {
            let inv_h_raw: &[u64] =
                unsafe { ext3_slice_to_u64::<E>(&inv_denoms_host[0..lde_size]) };
            let inv_t_raw: &[u64] = unsafe {
                ext3_slice_to_u64::<E>(&inv_denoms_host[lde_size..lde_size * (1 + num_eval_points)])
            };
            math_cuda::deep::deep_composition_ext3_with_dev_parts(
                main,
                aux_handle,
                parts,
                h_ood_raw,
                &trace_ood_raw,
                gammas_h_raw,
                &gammas_tr_raw,
                inv_h_raw,
                inv_t_raw,
                num_parts,
                num_main,
                num_aux,
                num_eval_points,
                row_stride_kernel,
                domain_size_kernel,
            )
        }
        (None, _) => {
            // De-interleave each ext3 part column into 3 contiguous base-field
            // slabs of length `lde_size` (the math-cuda kernel reads the parts
            // buffer with layout `h_lde[(p*3 + k) * lde_stride + r]`).
            let mut packed = vec![0u64; num_parts * 3 * lde_size];
            for (p, col) in parts_host.iter().enumerate() {
                let slice = unsafe { ext3_slice_to_u64::<E>(col) };
                for (r, chunk) in slice.chunks_exact(3).enumerate() {
                    packed[(p * 3) * lde_size + r] = chunk[0];
                    packed[(p * 3 + 1) * lde_size + r] = chunk[1];
                    packed[(p * 3 + 2) * lde_size + r] = chunk[2];
                }
            }
            parts_host_packed = packed;
            // Host inv_denoms required when going through this path; we
            // validated the slice length above.
            let inv_h_raw: &[u64] =
                unsafe { ext3_slice_to_u64::<E>(&inv_denoms_host[0..lde_size]) };
            let inv_t_raw: &[u64] = unsafe {
                ext3_slice_to_u64::<E>(&inv_denoms_host[lde_size..lde_size * (1 + num_eval_points)])
            };
            math_cuda::deep::deep_composition_ext3(
                main,
                aux_handle,
                &parts_host_packed,
                h_ood_raw,
                &trace_ood_raw,
                gammas_h_raw,
                &gammas_tr_raw,
                inv_h_raw,
                inv_t_raw,
                num_parts,
                num_main,
                num_aux,
                num_eval_points,
                row_stride_kernel,
                domain_size_kernel,
            )
        }
    };

    let deep_raw = match result {
        Ok(v) => v,
        Err(_) => return None,
    };
    GPU_DEEP_CALLS.fetch_add(1, Ordering::Relaxed);
    debug_assert_eq!(deep_raw.len(), lde_size * 3);
    Some(u64_to_ext3_vec::<E>(&deep_raw))
}

/// Build `inv_denoms[k*n + i] = 1 / (lift(coset_base[i]) - z_scalars[k])`
/// entirely on device. Used by both R3 OOD (n = trace_size, k_scalars =
/// num_eval_points) and R4 DEEP (n = lde_size, k_scalars = 1 +
/// num_eval_points). Returns a device handle the caller can slice and
/// thread into downstream dispatchers without ever D2H'ing the inverted
/// values; on type / threshold / cudarc failure returns `None` so the
/// caller can fall back to CPU `inplace_batch_inverse`.
///
/// The threshold check uses `gpu_lde_threshold()` against `n * k_scalars`,
/// matching the rest of the dispatch layer.
pub(crate) fn try_compute_and_invert_inv_denoms_dev<F, E>(
    coset_base: &[FieldElement<F>],
    z_scalars: &[FieldElement<E>],
    sign: math_cuda::inverse::DenomSign,
    stream: &Arc<CudaStream>,
) -> Option<CudaSlice<u64>>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let n = coset_base.len();
    let k_scalars = z_scalars.len();
    if n == 0 || k_scalars == 0 {
        return None;
    }
    let total = n.checked_mul(k_scalars)?;
    if total < gpu_lde_threshold() {
        return None;
    }

    // SAFETY: F == Goldilocks per TypeId check; FieldElement<F> is
    // #[repr(transparent)] over u64.
    let coset_u64: &[u64] = unsafe { from_raw_parts(coset_base.as_ptr() as *const u64, n) };
    let coset_dev = match stream.clone_htod(coset_u64) {
        Ok(s) => s,
        Err(_) => return None,
    };

    // SAFETY: E == Ext3 per TypeId check.
    let z_u64: &[u64] = unsafe { ext3_slice_to_u64::<E>(z_scalars) };

    let result = math_cuda::inverse::compute_and_invert_denoms_ext3_dev(
        &coset_dev, z_u64, n, k_scalars, sign, stream,
    );
    match result {
        Ok(handle) => {
            GPU_BATCH_INVERT_CALLS.fetch_add(1, Ordering::Relaxed);
            Some(handle)
        }
        Err(_) => None,
    }
}

/// Convenience wrapper for prover callers that don't yet own a stream:
/// acquires the math-cuda backend, allocates a fresh stream, and produces
/// a device-resident `inv_denoms` buffer plus the stream that owns it.
/// The caller passes the tuple through to the downstream dispatch
/// functions (`try_barycentric_*_on_handle`, `try_deep_composition_gpu`)
/// so every kernel touching the buffer runs on the same stream (no
/// cross-stream race).
///
/// Returns `None` on type / threshold mismatch, backend init failure, or
/// any cudarc error; the caller falls back to its CPU
/// `inplace_batch_inverse` loop.
pub(crate) fn try_inv_denoms_dev_with_stream<F, E>(
    coset_base: &[FieldElement<F>],
    z_scalars: &[FieldElement<E>],
    sign: math_cuda::inverse::DenomSign,
) -> Option<(CudaSlice<u64>, Arc<CudaStream>)>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    let be = math_cuda::device::backend().ok()?;
    let stream = be.next_stream();
    let handle =
        try_compute_and_invert_inv_denoms_dev::<F, E>(coset_base, z_scalars, sign, &stream)?;
    Some((handle, stream))
}

/// R3 OOD device-side context: bundles the inverted denominators, the
/// coset_points upload (used by every barycentric kernel for this batch),
/// and the stream so producer + consumers serialize naturally. Hoisting
/// `coset_points` here means the barycentric kernels read the same
/// device buffer across `num_eval_points * {main, aux}` calls instead
/// of re-uploading `dc.points` each iteration.
#[derive(Debug)]
pub(crate) struct R3DevContext {
    pub inv_denoms: CudaSlice<u64>,
    pub coset_points: CudaSlice<u64>,
    pub stream: Arc<CudaStream>,
}

/// Build an [`R3DevContext`] in one stream: acquire backend, allocate
/// stream, H2D coset_points once, then run `compute_and_invert_denoms`
/// against that same handle so the coset H2D isn't repeated by any
/// downstream barycentric kernel.
///
/// Returns `None` on type / threshold mismatch, backend init failure, or
/// any cudarc error.
pub(crate) fn try_prep_r3_dev_context<F, E>(
    coset_base: &[FieldElement<F>],
    z_scalars: &[FieldElement<E>],
) -> Option<R3DevContext>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let n = coset_base.len();
    let k_scalars = z_scalars.len();
    if n == 0 || k_scalars == 0 {
        return None;
    }
    let total = n.checked_mul(k_scalars)?;
    if total < gpu_lde_threshold() {
        return None;
    }

    let be = math_cuda::device::backend().ok()?;
    let stream = be.next_stream();

    // SAFETY: F == Goldilocks per TypeId check; FieldElement<F> is
    // #[repr(transparent)] over u64.
    let coset_u64: &[u64] = unsafe { from_raw_parts(coset_base.as_ptr() as *const u64, n) };
    let coset_points = stream.clone_htod(coset_u64).ok()?;

    // SAFETY: E == Ext3 per TypeId check.
    let z_u64: &[u64] = unsafe { ext3_slice_to_u64::<E>(z_scalars) };

    let inv_denoms = match math_cuda::inverse::compute_and_invert_denoms_ext3_dev(
        &coset_points,
        z_u64,
        n,
        k_scalars,
        math_cuda::inverse::DenomSign::ZMinusX,
        &stream,
    ) {
        Ok(h) => h,
        Err(_) => return None,
    };
    GPU_BATCH_INVERT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(R3DevContext {
        inv_denoms,
        coset_points,
        stream,
    })
}

/// R4 FRI dispatch: drive the full FRI commit phase device-side. Mirrors
/// [`crate::fri::commit_phase_from_evaluations`]: per-layer transcript
/// ping-pong (sample zeta, fold, build Merkle tree, append root).
///
/// Returns `None` on any failure (precondition miss or cudarc error
/// mid-loop). On a mid-loop failure the transcript is restored from the
/// snapshot taken at entry, so the caller's CPU fallback path runs against
/// a byte-identical pre-GPU transcript state and produces the same proof
/// it would have produced had the GPU never been tried. This requires the
/// concrete transcript type to support snapshot semantics via `Clone`.
#[allow(clippy::type_complexity)]
pub(crate) fn try_fri_commit_gpu<F, E, T>(
    number_layers: usize,
    evals: &[FieldElement<E>],
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> Option<(
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)>
where
    F: IsFFTField + IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
    T: IsStarkTranscript<E, F> + Clone,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let n0 = evals.len();
    if n0 != domain_size || !n0.is_power_of_two() || n0 < 2 {
        return None;
    }
    if n0 < gpu_lde_threshold() {
        return None;
    }

    // Pre-compute inv_twiddles on CPU (matches commit_phase_from_evaluations)
    // and pack to u64 before any transcript mutation, so on H2D / state
    // construction failure the caller's transcript is untouched.
    let inv_twiddles = compute_coset_twiddles_inv::<F>(coset_offset, domain_size);
    let mut inv_tw_u64: Vec<u64> = Vec::with_capacity(inv_twiddles.len());
    for t in &inv_twiddles {
        // SAFETY: F == Goldilocks per TypeId check; FieldElement<Gl> is
        // #[repr(transparent)] over u64.
        let v: u64 = unsafe { *(t.value() as *const _ as *const u64) };
        inv_tw_u64.push(v);
    }

    // SAFETY: E == Ext3; FieldElement<Ext3> backing is [u64; 3].
    let evals_u64: &[u64] = unsafe { ext3_slice_to_u64::<E>(evals) };

    let mut state = match math_cuda::fri::FriCommitState::new(evals_u64, &inv_tw_u64, n0) {
        Ok(s) => s,
        Err(_) => return None,
    };

    // Snapshot the transcript before any sampling. On a cudarc failure
    // mid-loop we restore from this snapshot and return None, so the CPU
    // fallback in `commit_phase_from_evaluations` starts from a byte-
    // identical transcript and produces the same proof it would have
    // produced had this dispatch never been called.
    let transcript_snapshot = transcript.clone();

    let num_committed_layers = number_layers.saturating_sub(1);
    let mut fri_layer_list: Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>> =
        Vec::with_capacity(num_committed_layers);

    for _ in 0..num_committed_layers {
        // <<<< Receive challenge zeta_k
        let zeta: FieldElement<E> = transcript.sample_field_element();
        // SAFETY: E == Ext3.
        let zeta_ptr = &zeta as *const FieldElement<E> as *const u64;
        let zeta_raw: [u64; 3] = unsafe { [*zeta_ptr, *zeta_ptr.add(1), *zeta_ptr.add(2)] };

        let (root, layer_evals_u64, nodes_bytes) = match state.fold_and_commit_layer(zeta_raw) {
            Ok(v) => v,
            Err(_) => {
                *transcript = transcript_snapshot.clone();
                return None;
            }
        };

        // Build the FriLayer: ext3 evals + Merkle tree from precomputed nodes.
        let evaluation = u64_to_ext3_vec::<E>(&layer_evals_u64);

        debug_assert!(nodes_bytes.len().is_multiple_of(32));
        let nodes: Vec<[u8; 32]> = nodes_bytes
            .chunks_exact(32)
            .map(|c| c.try_into().expect("chunks_exact(32) yields 32 bytes"))
            .collect();
        let merkle_tree = MerkleTree::<FriLayerMerkleTreeBackend<E>>::from_precomputed_nodes(nodes)
            .expect("FRI commit: precomputed nodes form a valid tree");

        fri_layer_list.push(FriLayer::new(&evaluation, merkle_tree));

        // >>>> Send commitment: [p_k]
        let mut root_arr = [0u8; 32];
        root_arr.copy_from_slice(&root);
        transcript.append_bytes(&root_arr);
    }

    // <<<< Receive challenge zeta_{n-1}
    let zeta_last: FieldElement<E> = transcript.sample_field_element();
    let zeta_ptr = &zeta_last as *const FieldElement<E> as *const u64;
    let zeta_raw: [u64; 3] = unsafe { [*zeta_ptr, *zeta_ptr.add(1), *zeta_ptr.add(2)] };

    let last_raw = match state.fold_final(zeta_raw) {
        Ok(v) => v,
        Err(_) => {
            *transcript = transcript_snapshot;
            return None;
        }
    };
    let last_vec = u64_to_ext3_vec::<E>(&last_raw);
    let last_value = last_vec
        .into_iter()
        .next()
        .expect("fold_final returns 1 elt");

    // >>>> Send value: p_n
    transcript.append_field_element(&last_value);

    GPU_FRI_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((last_value, fri_layer_list))
}
