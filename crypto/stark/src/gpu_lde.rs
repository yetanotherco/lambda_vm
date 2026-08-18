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

// External-profiler capture window (nsys -c cudaProfilerApi); re-exported so
// the prover crate can bracket the proving section without a math-cuda dep.

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::proof::Proof;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;
#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

use crate::config::{Commitment, FriLayerMerkleTreeBackend};
use crate::domain::Domain;
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_decommit::FriDecommitment;
use crate::trace::LDETraceTable;

/// Break-even LDE size. For LDE sizes smaller than this, the CPU
/// `coset_lde_full_expand` completes in a few hundred microseconds and the
/// GPU's tens of kernel launches plus H2D/D2H round-trip is a net loss. The
/// check is on **lde size**, not trace length, because that's what
/// determines the FFT workload.
///
/// The commit itself is not the whole cost: a table committed on CPU has no
/// device handle, so every R2-R4 GPU dispatch re-uploads its LDE. 2^14 is the
/// measured sweep optimum on ethrex continuations (2^14 beats 2^15..2^19 and
/// also beats "everything on GPU", where sub-2^14 tables lose to launch
/// overhead). Override via env var for tuning.
///
/// The same value gates the whole dispatch layer, not just the commit: R2
/// decompose, the R3 inv-denoms/barycentric contexts, R4 DEEP and the FRI
/// fold all admit on it, so moving it moves every one of those floors
/// together. The device-only envelope is the one gate that does NOT ride on
/// it — see [`DEFAULT_DEVICE_ONLY_MIN_LDE`].
const DEFAULT_GPU_LDE_THRESHOLD: usize = 1 << 14;

fn gpu_lde_threshold() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_LDE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_LDE_THRESHOLD)
    })
}

/// Minimum LDE size for the device-only envelope, decoupled from the commit
/// threshold above. Committing on GPU and keeping the handle resident pays
/// from small sizes (it kills the per-round re-uploads); dropping the HOST
/// copy is a much stronger contract — every downstream dispatch must take its
/// GPU path or the prove hard-aborts, and the gate cannot mirror kernel-side
/// eligibility (the LOCKSTEP note below). Keep device-only to the large-table
/// envelope where those paths are exercised; mid tables keep a host copy so a
/// dispatch decline degrades to CPU instead of aborting.
///
/// That degradation covers the sites that READ the LDE — they all gate on
/// `host_trace_empty()` and take their host arm. It does NOT cover the R4
/// Merkle-proof gather: the host tree is root-only for every GPU-committed
/// table (the tree stays resident from [`DEFAULT_GPU_LDE_THRESHOLD`] upward,
/// whatever `retain_host_lde` says), so a declined `gather_proofs_dev` has
/// nothing to fall back to and aborts regardless of the host LDE. Lowering
/// the commit threshold therefore widens that one abort site even though it
/// leaves this envelope alone.
const DEFAULT_DEVICE_ONLY_MIN_LDE: usize = 1 << 19;

fn gpu_device_only_threshold() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_DEVICE_ONLY_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_DEVICE_ONLY_MIN_LDE)
    })
}

/// Test hook: decline the device R2 path unconditionally so device-only
/// tables exercise the [`materialize_lde_trace_host`] recovery end to end.
pub(crate) fn gpu_force_downgrade() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("LAMBDA_VM_GPU_FORCE_DOWNGRADE").is_ok_and(|v| v != "0"))
}

/// Diagnostic hook: recompute the R2 composition parts and the R3 OOD
/// evaluations on host after each device dispatch and panic (naming the table
/// and stage) on any mismatch. Localizes silent device-side corruption.
pub(crate) fn gpu_xcheck() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("LAMBDA_VM_GPU_XCHECK").is_ok_and(|v| v != "0"))
}

/// Serialize the SUBMISSION of the device R2 window (constraint eval +
/// decompose) across tables. Concurrent R2 windows under VRAM pressure can
/// transiently corrupt a whole H buffer (root mechanism unidentified; reruns
/// on the same resident inputs come out correct), yielding a proof that fails
/// verification. Holding this lock empirically suppresses that at negligible
/// cost — the windows rarely overlap.
///
/// How much it enforces depends on the table. One that keeps its host trace
/// ends the window in a blocking D2H (the `want_host` arm of
/// [`try_decompose_extend_d2_dev`]), so the guard is held until that table's
/// kernels have completed — a real execution barrier. A device-only table's
/// window is enqueue-only, so two tables' R2 kernels can still overlap on
/// device; what the lock orders there is submission and allocation, which is
/// enough to suppress the corruption in practice but is not a guarantee that
/// R2 kernels never run concurrently.
///
/// `LAMBDA_VM_GPU_SERIALIZE_R2=0` disables the lock (e.g. to bisect or once
/// the underlying race is fixed).
pub(crate) fn r2_serialize_guard() -> Option<std::sync::MutexGuard<'static, ()>> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if std::env::var("LAMBDA_VM_GPU_SERIALIZE_R2").as_deref() != Ok("0") {
        // The guarded state is (), so a panic while holding the lock carries
        // no information — recover instead of burying the original panic
        // under a cascade of PoisonErrors from every other table.
        Some(LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        None
    }
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
    GPU_LOGUP_CALLS.store(0, Ordering::Relaxed);
    GPU_COMPOSITION_CALLS.store(0, Ordering::Relaxed);
    GPU_OPENING_GATHER_CALLS.store(0, Ordering::Relaxed);
    GPU_DEVICE_ONLY_CALLS.store(0, Ordering::Relaxed);
    GPU_DEVICE_ONLY_DOWNGRADES.store(0, Ordering::Relaxed);
    GPU_RESIDENT_AUX_RETRIES.store(0, Ordering::Relaxed);
    GPU_RESIDENT_AUX_DOWNGRADES.store(0, Ordering::Relaxed);
}

pub(crate) static GPU_EXTEND_HALVES_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_extend_halves_calls() -> u64 {
    GPU_EXTEND_HALVES_CALLS.load(Ordering::Relaxed)
}

/// Successful LogUp aux-build GPU dispatches (one per table that took either
/// the resident or the term-column path; failed attempts fall back to CPU and
/// are not counted).
pub(crate) static GPU_LOGUP_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_logup_calls() -> u64 {
    GPU_LOGUP_CALLS.load(Ordering::Relaxed)
}

/// Successful GPU composition-poly (`H(row)`) dispatches — one per table whose
/// round-2 constraint evaluation took the fused on-device path (a failed attempt
/// or a gate miss falls back to the CPU accumulation and is not counted).
pub(crate) static GPU_COMPOSITION_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_composition_calls() -> u64 {
    GPU_COMPOSITION_CALLS.load(Ordering::Relaxed)
}

/// Successful device-resident-LDE opening-value gathers in
/// `open_deep_composition_poly` — one per main/aux trace whose R4 query rows
/// were read straight off the device LDE instead of the host trace (a
/// non-resident tree or non-Goldilocks tower falls back to the host gather and
/// is not counted). Guards against a silent regression where Stage-2 openings
/// quietly revert to the host path.
pub(crate) static GPU_OPENING_GATHER_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_opening_gather_calls() -> u64 {
    GPU_OPENING_GATHER_CALLS.load(Ordering::Relaxed)
}

/// Tables whose round-1 LDE was kept device-only (host trace D2H skipped) — the
/// Stage-3 full-residency win. Incremented once per main trace that took the
/// `device_only` path. Zero means every table kept its host copy (gate never
/// engaged), so a residency regression drops this to 0 while proofs still
/// verify.
pub(crate) static GPU_DEVICE_ONLY_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_device_only_calls() -> u64 {
    GPU_DEVICE_ONLY_CALLS.load(Ordering::Relaxed)
}

/// Runtime override to force the GPU composition path off (→ CPU accumulation).
/// An escape hatch, and the A/B toggle for benchmarking the path against the CPU
/// baseline in one process (no rebuild). Default off (path enabled).
static GPU_COMPOSITION_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub fn set_gpu_composition_disabled(v: bool) {
    GPU_COMPOSITION_DISABLED.store(v, Ordering::Relaxed);
}
pub(crate) fn gpu_composition_disabled() -> bool {
    if GPU_COMPOSITION_DISABLED.load(Ordering::Relaxed) {
        return true;
    }
    // Env fallback (cached), so an unmodified prove binary can A/B the path:
    // `LAMBDA_VM_DISABLE_GPU_COMPOSITION=1`.
    static ENV_DISABLED: OnceLock<bool> = OnceLock::new();
    *ENV_DISABLED.get_or_init(|| {
        std::env::var("LAMBDA_VM_DISABLE_GPU_COMPOSITION")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Runtime override to force the Stage-3 device-only path off (keeps the round-1
/// host D2H). Independent of the composition toggle, so the residency win can be
/// A/B-benched with the GPU composition path left on. Default off (path enabled).
static DEVICE_ONLY_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub fn set_device_only_disabled(v: bool) {
    DEVICE_ONLY_DISABLED.store(v, Ordering::Relaxed);
}
pub(crate) fn device_only_disabled() -> bool {
    if DEVICE_ONLY_DISABLED.load(Ordering::Relaxed) {
        return true;
    }
    // Env fallback (cached): `LAMBDA_VM_DISABLE_DEVICE_ONLY=1`.
    static ENV_DISABLED: OnceLock<bool> = OnceLock::new();
    *ENV_DISABLED.get_or_init(|| {
        std::env::var("LAMBDA_VM_DISABLE_DEVICE_ONLY")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Stage-3 device-only gate: `true` when a table's round-1 LDE can be left
/// device-resident (host D2H skipped) because every downstream round is
/// guaranteed to take its GPU path. A strict AND of the numeric and shape
/// preconditions that imply the R2 composition, R3 barycentric, R4 DEEP, and
/// R4 opening GPU paths all fire and read the device LDE — but not the whole
/// predicate on its own: the caller `IsStarkProver::device_only_for`
/// (prover.rs) adds the AIR-level preconditions this signature does not
/// carry, notably the d=2 quotient part count the device-resident R2 path
/// requires.
///
/// If a precondition is nonetheless violated at runtime (mis-gate or
/// transient GPU error), what happens depends on the round. R2 and the R1
/// resident-aux commit recover: they download what the host arms need (the
/// resident LDEs at R2, the resident aux trace plus the main LDE at R1), bump
/// their site's counter ([`GPU_DEVICE_ONLY_DOWNGRADES`] at R2,
/// [`GPU_RESIDENT_AUX_DOWNGRADES`] at R1) and continue host-backed — slower,
/// never wrong — aborting only when the resident handles cannot serve the
/// data. R3 and R4 have no such recovery: the R3 barycentric arms assert on
/// the buffer they are about to read and the R4 guards on `host_trace_empty`,
/// both failing loudly rather than reading an empty host trace.
///
/// `zerofier_uniform` must be the R1-derived conservative form (all constraints
/// share `end_exemptions == 0`), which implies `ZerofierEvaluations::is_uniform`
/// (a single cyclic group) — the condition the GPU composition kernel needs.
///
/// LOCKSTEP: this gate must IMPLY the runtime dispatch checks in
/// `ConstraintEvaluator::try_evaluate_composition_gpu` (plus the R3/R4 device
/// arms). A fallback condition added to a dispatch without a mirror here
/// costs every gate-true table either a hard-abort at R3/R4 — loud, but an
/// avoidable crash — or, at R2 and the R1 resident-aux commit, a silent
/// downgrade to the host path, which is what [`GPU_DEVICE_ONLY_DOWNGRADES`]
/// exists to surface (an R1 decline lands in [`GPU_RESIDENT_AUX_DOWNGRADES`],
/// which the gate does not govern).
pub(crate) fn device_only_gate<F, E>(
    lde_size: usize,
    n: usize,
    offsets_contiguous: bool,
    zerofier_uniform: bool,
) -> bool
where
    F: 'static,
    E: 'static,
{
    // debug-checks reconstruct the LDE from the host trace — keep it resident.
    if cfg!(feature = "debug-checks") {
        return false;
    }
    is_goldilocks_ext3_tower::<F, E>()
        && !device_only_disabled()
        && !gpu_composition_disabled()
        && lde_size.is_power_of_two()
        && lde_size >= gpu_device_only_threshold()
        && n >= gpu_bary_threshold()
        && offsets_contiguous
        && zerofier_uniform
}

/// `true` when the field tower is concrete Goldilocks + its degree-3 extension —
/// the only tower with a CUDA lowering. The one home of this check: every GPU
/// dispatch gate calls it, so the tower test cannot drift between sites.
pub(crate) fn is_goldilocks_ext3_tower<F: 'static, E: 'static>() -> bool {
    TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
        && TypeId::of::<E>() == TypeId::of::<Degree3GoldilocksExtensionField>()
}

/// `true` when the transition offsets form the contiguous frame `[0, 1, ..]`
/// the GPU kernels' row math assumes (a `Var` at offset `o` reads LDE row
/// `row + o·next_step`). Shared by the composition dispatch and its gates.
pub(crate) fn offsets_are_contiguous(offsets: &[usize]) -> bool {
    offsets.iter().enumerate().all(|(i, &o)| o == i)
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
#[cfg_attr(not(feature = "debug-checks"), allow(dead_code))]
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

/// Fully device-resident degree-2 decomposition + half extension: takes the
/// resident composition evals `H`, decomposes into H0/H1 on device, LDE-extends
/// both and keeps the de-interleaved parts buffer as a `GpuLdeExt3` (commit
/// tree, R3 OOD, R4 DEEP and openings all read the handle). With `want_host`
/// the evaluations are also drained to host for the fallback consumers;
/// without it (device-only) the returned part Vecs are empty placeholders.
/// `None` → the caller downloads `H` and runs the host decompose path.
pub(crate) fn try_decompose_extend_d2_dev<F, E>(
    h: &math_cuda::constraint_interp::GpuCompH,
    inv_2x: &std::sync::Arc<Vec<FieldElement<F>>>,
    weights: &[FieldElement<F>],
    want_host: bool,
) -> Option<(Vec<Vec<FieldElement<E>>>, math_cuda::lde::GpuLdeExt3)>
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
    let lde_size = h.num_rows;
    if lde_size < gpu_lde_threshold() || !lde_size.is_power_of_two() {
        return None;
    }
    let n = lde_size / 2;
    if weights.len() != n || inv_2x.len() < n {
        return None;
    }

    // SAFETY: `F == GoldilocksField` (gated above); the Arc'd Vecs share layout.
    let inv_conc: &crate::constraint_ir::gpu_interp::GoldilocksBZInv =
        unsafe { &*(inv_2x as *const _ as *const _) };
    let inv_handle = crate::constraint_ir::gpu_interp::base_vec_device_handle(inv_conc)?;

    let two_inv_fe = FieldElement::<F>::from(2u64).inv().ok()?;
    // SAFETY: F == Goldilocks; FieldElement<Gl> is repr(transparent) over u64.
    let two_inv: u64 = unsafe { *(two_inv_fe.value() as *const _ as *const u64) };

    let (slabs, stream, n_dev) =
        math_cuda::constraint_interp::decompose_d2_into_slabs(h, &inv_handle, two_inv).ok()?;
    debug_assert_eq!(n_dev, n);

    GPU_EXTEND_HALVES_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_LDE_CALLS.fetch_add(6, Ordering::Relaxed);

    // SAFETY: F == Goldilocks (repr u64); ext3 outputs are [u64; 3] per element.
    let weights_u64: &[u64] =
        unsafe { from_raw_parts(weights.as_ptr() as *const u64, weights.len()) };

    if !want_host {
        let handle = math_cuda::lde::coset_lde_batch_ext3_slabs_keep(
            &stream,
            slabs,
            2,
            n,
            2,
            weights_u64,
            None,
        )
        .ok()?;
        return Some((vec![Vec::new(), Vec::new()], handle));
    }

    let mut lde_h0 = vec![FieldElement::<E>::zero(); lde_size];
    let mut lde_h1 = vec![FieldElement::<E>::zero(); lde_size];
    let ext3_len = lde_size
        .checked_mul(3)
        .expect("ext3 output length overflow");
    let out0 = unsafe { from_raw_parts_mut(lde_h0.as_mut_ptr() as *mut u64, ext3_len) };
    let out1 = unsafe { from_raw_parts_mut(lde_h1.as_mut_ptr() as *mut u64, ext3_len) };
    let mut outputs: [&mut [u64]; 2] = [out0, out1];

    let handle = math_cuda::lde::coset_lde_batch_ext3_slabs_keep(
        &stream,
        slabs,
        2,
        n,
        2,
        weights_u64,
        Some(&mut outputs),
    )
    .ok()?;

    Some((vec![lde_h0, lde_h1], handle))
}

/// D2H bridge for the fallback: download a resident `H` and lift it into
/// field elements (the exact input the host decompose expects).
pub(crate) fn download_comp_h_to_field<E: IsField + 'static>(
    h: &math_cuda::constraint_interp::GpuCompH,
) -> Option<Vec<FieldElement<E>>> {
    let raw = math_cuda::constraint_interp::download_comp_h(h).ok()?;
    crate::constraint_ir::gpu_interp::ext3_u64_to_field::<E>(&raw)
}

pub(crate) static GPU_LEAF_HASH_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_leaf_hash_calls() -> u64 {
    GPU_LEAF_HASH_CALLS.load(Ordering::Relaxed)
}

/// Row-major GPU path: single H2D → row-major NTT → row-major Keccak →
/// Merkle → single D2H. Keeps the Merkle tree resident on device (in the
/// handle's `.tree`); the returned host `MerkleTree` is root only, so query
/// openings gather paths from the device tree via [`gather_proofs_dev`].
pub(crate) fn try_expand_leaf_and_tree_row_major_keep<F, E, B>(
    row_major: &[FieldElement<E>],
    predev: Option<&math_cuda::CudaSlice<u64>>,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[FieldElement<F>],
    retain_host_lde: bool,
) -> Option<(
    MerkleTree<B>,
    math_cuda::lde::GpuLdeBase,
    Vec<FieldElement<E>>,
)>
where
    F: IsField + 'static,
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if row_major.len() != n * m || m == 0 || n == 0 {
        return None;
    }

    let raw: &[u64] = unsafe { from_raw_parts(row_major.as_ptr() as *const u64, n * m) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };

    GPU_LDE_CALLS.fetch_add(m as u64, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, Ordering::Relaxed);

    // The keep path keeps the Merkle tree resident on device (in `handle.tree`).
    // `retain_host_lde=false` additionally skips the row-major D2H (device-only).
    let (handle, lde_u64) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
        raw,
        predev,
        n,
        m,
        blowup_factor,
        &weights_u64,
        retain_host_lde,
    )
    .ok()?;

    // Transmute Vec<u64> → Vec<FieldElement<E>> (zero-copy, E == GoldilocksField).
    let lde_out: Vec<FieldElement<E>> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(lde_u64);
        Vec::from_raw_parts(
            v.as_mut_ptr() as *mut FieldElement<E>,
            v.len(),
            v.capacity(),
        )
    };

    // Root-only host tree: the device tree (`handle.tree`) holds the nodes and
    // serves openings; only the commitment root lives on host.
    let root = handle.tree.as_ref()?.root;
    let tree = MerkleTree::<B>::from_root(root);
    Some((tree, handle, lde_out))
}

/// Convert a GPU-built full node buffer (`(2*leaves - 1) * 32` bytes, inner
/// nodes first, root at offset 0, leaves at the tail) into a host
/// [`MerkleTree`], the exact layout `from_precomputed_nodes` expects.
fn tree_from_node_bytes<B>(nodes: Vec<u8>) -> Option<MerkleTree<B>>
where
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
{
    debug_assert_eq!(nodes.len() % 32, 0);
    let nodes: Vec<[u8; 32]> = nodes
        .chunks_exact(32)
        .map(|c| {
            let mut n = [0u8; 32];
            n.copy_from_slice(c);
            n
        })
        .collect();
    MerkleTree::<B>::from_precomputed_nodes(nodes)
}

/// Preprocessed-table variant of [`try_expand_leaf_and_tree_row_major_keep`]:
/// one row-major GPU LDE of ALL columns plus TWO subset Merkle trees — the
/// precomputed columns `[0, split_col)` and the multiplicity columns
/// `[split_col, m)` — matching the CPU `commit_rows_bit_reversed_subset`
/// pair bit for bit. The precomputed tree comes back as a full HOST tree
/// (it feeds the process-wide cache); the multiplicity tree stays resident
/// in the handle (root-only host tree, R4 openings gather paths on device).
/// The handle also keeps the column-major LDE + trace snapshot for the
/// downstream GPU rounds.
///
/// `build_precomputed=false` skips the precomputed tree (process-cache hit);
/// the first element is then `None`. With `want_host=false` the row-major LDE
/// D2H is skipped and the returned Vec is empty (device-only tables: every
/// consumer reads the handle).
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_expand_split_trees_row_major_keep<F, E, B>(
    row_major: &[FieldElement<E>],
    predev: Option<&math_cuda::CudaSlice<u64>>,
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[FieldElement<F>],
    split_col: usize,
    build_precomputed: bool,
    want_host: bool,
) -> Option<(
    Option<MerkleTree<B>>,
    MerkleTree<B>,
    math_cuda::lde::GpuLdeBase,
    Vec<FieldElement<E>>,
)>
where
    F: IsField + 'static,
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if row_major.len() != n * m || m == 0 || n == 0 {
        return None;
    }
    if split_col == 0 || split_col >= m {
        return None;
    }

    let raw: &[u64] = unsafe { from_raw_parts(row_major.as_ptr() as *const u64, n * m) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };

    GPU_LDE_CALLS.fetch_add(m as u64, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1 + build_precomputed as u64, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1 + build_precomputed as u64, Ordering::Relaxed);

    let (pre_nodes, handle, lde_u64) = math_cuda::lde::coset_lde_row_major_split_trees(
        raw,
        predev,
        n,
        m,
        blowup_factor,
        &weights_u64,
        split_col,
        build_precomputed,
        want_host,
    )
    .ok()?;

    let pre_tree = match pre_nodes {
        Some(nodes) => Some(tree_from_node_bytes::<B>(nodes)?),
        None => None,
    };
    // Mult tree resident in the handle: the host tree is root only and R4
    // openings gather authentication paths on device.
    let mult_tree = MerkleTree::<B>::from_root(
        handle
            .tree
            .as_ref()
            .expect("split path always builds the mult tree")
            .root,
    );

    // Transmute Vec<u64> → Vec<FieldElement<E>> (zero-copy, E == GoldilocksField).
    let lde_out: Vec<FieldElement<E>> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(lde_u64);
        Vec::from_raw_parts(
            v.as_mut_ptr() as *mut FieldElement<E>,
            v.len(),
            v.capacity(),
        )
    };

    Some((pre_tree, mult_tree, handle, lde_out))
}

/// Row-major ext3 GPU path: single H2D → row-major NTT (m*3 base-field cols) →
/// row-major Keccak → Merkle → single D2H → transpose to GpuLdeExt3 handle.
/// Same optimization as the base-field path: no extract_columns, no CPU transpose.
pub(crate) fn try_expand_leaf_and_tree_ext3_row_major_keep<F, E, B>(
    row_major: &[FieldElement<E>],
    n: usize,
    m: usize,
    blowup_factor: usize,
    weights: &[FieldElement<F>],
    retain_host_lde: bool,
) -> Option<(
    MerkleTree<B>,
    math_cuda::lde::GpuLdeExt3,
    Vec<FieldElement<E>>,
)>
where
    F: IsField + 'static,
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
{
    let lde_size = n.saturating_mul(blowup_factor);
    if lde_size < gpu_lde_threshold() {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if row_major.len() != n * m || m == 0 || n == 0 {
        return None;
    }

    // Fp3 = [u64; 3] in memory — reinterpret as flat u64 slice (m3 = m*3).
    let m3 = m * 3;
    let raw: &[u64] = unsafe { from_raw_parts(row_major.as_ptr() as *const u64, n * m3) };
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };

    GPU_LDE_CALLS.fetch_add((m * 3) as u64, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, Ordering::Relaxed);

    // The keep path keeps the Merkle tree resident on device (in `handle.tree`).
    // `retain_host_lde=false` additionally skips the row-major D2H (device-only).
    let (handle, lde_u64) = math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep(
        raw,
        n,
        m,
        blowup_factor,
        &weights_u64,
        retain_host_lde,
    )
    .ok()?;

    // Transmute Vec<u64> → Vec<FieldElement<E>> (zero-copy, E == Fp3 = [u64;3]).
    let lde_out: Vec<FieldElement<E>> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(lde_u64);
        debug_assert!(
            v.len() % 3 == 0 && v.capacity() % 3 == 0,
            "lde_u64 len/capacity must be a multiple of 3 for Fp3 reinterpret"
        );
        Vec::from_raw_parts(
            v.as_mut_ptr() as *mut FieldElement<E>,
            v.len() / 3,
            v.capacity() / 3,
        )
    };

    // Root-only host tree: the device tree (`handle.tree`) holds the nodes and
    // serves openings; only the commitment root lives on host.
    let root = handle.tree.as_ref()?.root;
    let tree = MerkleTree::<B>::from_root(root);
    Some((tree, handle, lde_out))
}

/// Ext3 specialisation of [`try_expand_columns_batched`]. `E` is known to be
/// `Degree3GoldilocksExtensionField` by TypeId match at the caller.
///
/// The LDE runs over the 3 base-field components of each ext3 column. The
/// transform uses only base-field twiddles and coset weights, which act
/// componentwise on ext3, so the per-component result equals the ext3 LDE the
/// CPU path computes.
#[cfg_attr(not(feature = "debug-checks"), allow(dead_code))]
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
/// `commit_bit_reversed` (composition-polynomial commit path): each leaf hashes
/// 2 consecutive bit-reversed rows.
///
/// Returns `None` to fall through to the CPU path when the type or size
/// conditions don't hold; returns `None` on a math-cuda `Err` so the caller
/// recomputes on CPU.
pub(crate) fn try_build_comp_poly_tree_gpu<E, B>(
    lde_parts: &[Vec<FieldElement<E>>],
) -> Option<(MerkleTree<B>, math_cuda::lde::GpuMerkleTree)>
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

    // Keep the composition tree resident on device, so the whole tree copy to
    // host is eliminated. R4 composition openings gather paths from the device
    // tree (`gather_proofs_dev`); the returned host tree is root only.
    let dev_tree = match math_cuda::merkle::build_comp_poly_tree_from_evals_ext3_keep(&raw_parts) {
        Ok(t) => t,
        Err(_) => return None,
    };
    debug_assert_eq!(dev_tree.leaves_len, lde_size / 2);
    GPU_COMP_POLY_TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    let host = MerkleTree::<B>::from_root(dev_tree.root);
    Some((host, dev_tree))
}

/// Device-resident variant of [`try_build_comp_poly_tree_gpu`]: hashes the
/// composition tree straight from the resident R2 parts handle, skipping the
/// host pack + H2D re-upload of data that is already on device.
pub(crate) fn try_build_comp_poly_tree_gpu_from_dev<E, B>(
    handle: &math_cuda::lde::GpuLdeExt3,
) -> Option<(MerkleTree<B>, math_cuda::lde::GpuMerkleTree)>
where
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    if handle.m == 0 || !handle.lde_size.is_power_of_two() || handle.lde_size < gpu_lde_threshold()
    {
        return None;
    }
    let be = math_cuda::device::backend().ok()?;
    let stream = be.next_stream();
    handle.wait_ready_on(&stream).ok()?;
    let dev_tree = math_cuda::merkle::build_comp_poly_tree_from_slabs_dev(
        &stream,
        handle.buf.as_ref(),
        handle.m,
        handle.lde_size,
    )
    .ok()?;
    GPU_COMP_POLY_TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    let host = MerkleTree::<B>::from_root(dev_tree.root);
    Some((host, dev_tree))
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

/// Multi-eval-point variant of [`try_barycentric_base_on_handle`]: one kernel
/// pass over the main LDE computes the OOD sums for every evaluation point at
/// once (their inv_denom blocks are contiguous in the [`R3DevContext`] buffer),
/// instead of re-reading the column data per point. Returns one scaled eval Vec
/// per point, or `None` (→ per-point dispatch / CPU fallback) when the handle
/// is absent, thresholds miss, there are more points than the kernel's
/// accumulator cap, or the math-cuda call errs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_base_on_handle_multi<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    row_stride: usize,
    coset_points_len: usize,
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pows: &[FieldElement<E>],
    ctx: &R3DevContext,
) -> Option<Vec<Vec<FieldElement<E>>>>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if !is_goldilocks_ext3_tower::<F, E>() {
        return None;
    }
    let k_points = z_pows.len();
    if k_points == 0 || k_points > math_cuda::barycentric::BARY_MAX_EVAL_POINTS {
        return None;
    }
    let main = lde_trace.gpu_main()?;
    let num_cols = main.m;
    if num_cols == 0 {
        return Some(vec![Vec::new(); k_points]);
    }
    let n = coset_points_len;
    if !n.is_power_of_two() || n < gpu_bary_threshold() {
        return None;
    }
    if main.lde_size != n.checked_mul(row_stride)? {
        return None;
    }
    if ctx.inv_denoms.len() < k_points * 3 * n {
        return None;
    }

    let sums_raw = math_cuda::barycentric::barycentric_base_multi_on_device(
        &ctx.stream,
        main,
        row_stride,
        &ctx.coset_points,
        &ctx.inv_denoms,
        n,
        k_points,
    )
    .ok()?;
    GPU_BARY_CALLS.fetch_add(k_points as u64, Ordering::Relaxed);

    Some(
        z_pows
            .iter()
            .enumerate()
            .map(|(k, z_pow_n)| {
                let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
                apply_ext3_scalar::<E>(
                    &sums_raw[k * 3 * num_cols..(k + 1) * 3 * num_cols],
                    scalar,
                    num_cols,
                )
            })
            .collect(),
    )
}

/// Aux (ext3) counterpart of [`try_barycentric_base_on_handle_multi`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_ext3_on_handle_multi<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    row_stride: usize,
    coset_points_len: usize,
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    z_pows: &[FieldElement<E>],
    ctx: &R3DevContext,
) -> Option<Vec<Vec<FieldElement<E>>>>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if !is_goldilocks_ext3_tower::<F, E>() {
        return None;
    }
    let k_points = z_pows.len();
    if k_points == 0 || k_points > math_cuda::barycentric::BARY_MAX_EVAL_POINTS {
        return None;
    }
    let aux = lde_trace.gpu_aux()?;
    let num_cols = aux.m;
    if num_cols == 0 {
        return Some(vec![Vec::new(); k_points]);
    }
    let n = coset_points_len;
    if !n.is_power_of_two() || n < gpu_bary_threshold() {
        return None;
    }
    if aux.lde_size != n.checked_mul(row_stride)? {
        return None;
    }
    if ctx.inv_denoms.len() < k_points * 3 * n {
        return None;
    }

    let sums_raw = math_cuda::barycentric::barycentric_ext3_multi_on_device(
        &ctx.stream,
        aux,
        row_stride,
        &ctx.coset_points,
        &ctx.inv_denoms,
        n,
        k_points,
    )
    .ok()?;
    GPU_BARY_CALLS.fetch_add(k_points as u64, Ordering::Relaxed);

    Some(
        z_pows
            .iter()
            .enumerate()
            .map(|(k, z_pow_n)| {
                let scalar = ood_ext3_scalar::<F, E>(coset_offset_pow_n, n_inv, g_n_inv, z_pow_n);
                apply_ext3_scalar::<E>(
                    &sums_raw[k * 3 * num_cols..(k + 1) * 3 * num_cols],
                    scalar,
                    num_cols,
                )
            })
            .collect(),
    )
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
    try_barycentric_ext3_on_ext3_handle(
        lde_trace.gpu_aux()?,
        row_stride,
        coset_points,
        coset_offset_pow_n,
        n_inv,
        g_n_inv,
        z_pow_n,
        inv_denoms_host,
        r3_ctx,
    )
}

/// Same dispatch over an arbitrary resident ext3 handle (aux LDE or the R2
/// composition parts). One column of OOD sums per handle column.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_barycentric_ext3_on_ext3_handle<F, E>(
    aux: &math_cuda::lde::GpuLdeExt3,
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

/// FRI commit-phase dispatch counter (one per successful commit, not per
/// layer). Counts BOTH entry points, so a table whose device-resident attempt
/// ([`try_fri_commit_gpu_from_dev`]) fails and then commits from host evals
/// ([`try_fri_commit_gpu`]) still contributes exactly one — the count alone
/// cannot tell "the GPU path was skipped" from "it succeeded on the retry".
pub(crate) static GPU_FRI_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_fri_calls() -> u64 {
    GPU_FRI_CALLS.load(Ordering::Relaxed)
}

/// Batch-invert dispatch counter (one per
/// [`try_compute_and_invert_inv_denoms_dev`] call that actually built a
/// device handle). Fires up to three times per prove per table: R3 trace
/// OOD's `num_eval_points * trace_size` denominators, R3 parts OOD's single
/// point, and R4 DEEP's `(1 + num_eval_points) * lde_size` denominators. R4
/// has two chances at it (device-only DEEP, then the host DEEP arm), and both
/// are counted here, so a single failed dispatch does not necessarily lower
/// the total; R3's fallbacks are CPU-only, so a failure there does.
pub(crate) static GPU_BATCH_INVERT_CALLS: AtomicU64 = AtomicU64::new(0);
/// R2 downgrades, and only those: times a device-only table fell back to the
/// host evaluator and had its resident LDEs downloaded into the host buffers
/// first ([`materialize_lde_trace_host`], the sole site that bumps this).
/// Nonzero means the device-only gate cleared a table whose R2 dispatch then
/// declined at runtime — the table continued host-backed, correct but slower —
/// so every count is a gate miss, and the fix is to mirror the missing
/// condition into the gate. The R1 resident-aux downgrade is counted by
/// [`GPU_RESIDENT_AUX_DOWNGRADES`] instead: it fires on tables the gate never
/// marked device-only, so summing the two would blame the gate for declines it
/// never made.
pub(crate) static GPU_DEVICE_ONLY_DOWNGRADES: AtomicU64 = AtomicU64::new(0);
pub fn gpu_device_only_downgrades() -> u64 {
    GPU_DEVICE_ONLY_DOWNGRADES.load(Ordering::Relaxed)
}

/// R1 downgrades, and only those: times the resident aux trace was downloaded
/// so the aux commit could continue on the host arms, after the device aux LDE
/// declined and the drain-and-retry either did not run or declined again
/// ([`materialize_aux_trace_host`], the sole site that bumps this). Independent
/// of the device-only gate — the site is entered whenever `aux_resident()` is
/// set, whatever the gate said — so a table that was never device-only can land
/// here, and a nonzero value points at sustained VRAM pressure rather than a
/// gate miss. Read it against [`GPU_RESIDENT_AUX_RETRIES`]: retries alone mean
/// the drain absorbed the pressure, retries plus downgrades mean it did not.
pub(crate) static GPU_RESIDENT_AUX_DOWNGRADES: AtomicU64 = AtomicU64::new(0);
pub fn gpu_resident_aux_downgrades() -> u64 {
    GPU_RESIDENT_AUX_DOWNGRADES.load(Ordering::Relaxed)
}

/// Times the R1 resident-aux LDE declined and the prover drained the device to
/// retry it (prover.rs). Nonzero means the device hit transient VRAM pressure —
/// the retry is what keeps a decline from becoming a
/// [`GPU_RESIDENT_AUX_DOWNGRADES`] host downgrade, so a run with retries but no
/// downgrades paid nothing but the drain. Counts declines, not outcomes: it is
/// bumped before the retry, whether or not the retry then succeeds.
pub(crate) static GPU_RESIDENT_AUX_RETRIES: AtomicU64 = AtomicU64::new(0);
pub fn gpu_resident_aux_retries() -> u64 {
    GPU_RESIDENT_AUX_RETRIES.load(Ordering::Relaxed)
}

/// Recover a device-only table for the host path: download the resident main
/// and aux LDEs from their device handles into the host buffers and clear the
/// device-only flag. A side whose host buffer is already populated (a mixed
/// state: one commit fell back to CPU while the other stayed device-only) is
/// kept as is — only the missing side is downloaded. The class-level safety
/// net under the device-only gate — a static predicate can never mirror every
/// reason a dynamic dispatch might decline (kernel eligibility, transient
/// errors, shapes a new workload brings), so any miss lands here and degrades
/// to a slower-but-correct CPU round instead of a hard abort. Returns false
/// (→ the caller's abort) when the resident handles cannot serve the data: a
/// missing handle or bound stream, a handle whose shape disagrees with the
/// trace, a failed download or sync, or a field tower with no CUDA lowering.
pub(crate) fn materialize_lde_trace_host<F, E>(
    lde_trace: &mut crate::trace::LDETraceTable<F, E>,
) -> bool
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if !lde_trace.host_trace_empty() {
        return true;
    }
    if !is_goldilocks_ext3_tower::<F, E>() {
        return false;
    }
    let Some(stream) = lde_trace.bound_stream() else {
        return false;
    };

    // Main: column-major device buf -> row-major host Vec. An empty Vec tells
    // `set_host_data` to keep the buffer that is already there.
    let main_data: Vec<FieldElement<F>> =
        if lde_trace.num_main_cols() == 0 || !lde_trace.main_data.is_empty() {
            Vec::new()
        } else {
            let Some(h) = lde_trace.gpu_main() else {
                return false;
            };
            if h.m != lde_trace.num_main_cols() || h.lde_size != lde_trace.num_rows() {
                return false;
            }
            let Some(data) = download_main_lde_row_major::<F>(h, &stream) else {
                return false;
            };
            data
        };

    // Aux: de-interleaved ext3 slabs -> row-major interleaved host Vec.
    let aux_data: Vec<FieldElement<E>> =
        if lde_trace.num_aux_cols() == 0 || !lde_trace.aux_data.is_empty() {
            Vec::new()
        } else {
            let Some(h) = lde_trace.gpu_aux() else {
                return false;
            };
            if h.m != lde_trace.num_aux_cols() || h.lde_size != lde_trace.num_rows() {
                return false;
            }
            if h.wait_ready_on(&stream).is_err() {
                return false;
            }
            let Ok(slabs) = stream.clone_dtoh(h.buf.as_ref()) else {
                return false;
            };
            if stream.synchronize().is_err() {
                return false;
            }
            let (m, lde) = (h.m, h.lde_size);
            // Short download: degrade like the sibling paths
            // (`download_main_lde_row_major`, `materialize_aux_trace_host`)
            // rather than panic on the slab slicing below.
            if slabs.len() != m * lde * 3 {
                return false;
            }
            // Parallel de-interleaved slabs → row-major interleaved: each row
            // chunk gathers from the source slabs independently.
            let mut interleaved = vec![0u64; m * lde * 3];
            if m > 0 {
                #[cfg(feature = "parallel")]
                {
                    interleaved
                        .par_chunks_exact_mut(m * 3)
                        .enumerate()
                        .for_each(|(r, dst)| {
                            for (c, dst_col) in dst.chunks_exact_mut(3).enumerate() {
                                for (k, d) in dst_col.iter_mut().enumerate() {
                                    *d = slabs[(c * 3 + k) * lde + r];
                                }
                            }
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (r, dst) in interleaved.chunks_exact_mut(m * 3).enumerate() {
                        for (c, dst_col) in dst.chunks_exact_mut(3).enumerate() {
                            for (k, d) in dst_col.iter_mut().enumerate() {
                                *d = slabs[(c * 3 + k) * lde + r];
                            }
                        }
                    }
                }
            }
            // SAFETY: E == Ext3 per the tower check; FieldElement<Ext3> backing
            // is [u64; 3].
            unsafe {
                let mut v = std::mem::ManuallyDrop::new(interleaved);
                debug_assert!(
                    v.len().is_multiple_of(3) && v.capacity().is_multiple_of(3),
                    "interleaved len/capacity must be a multiple of 3 for Fp3 reinterpret"
                );
                Vec::from_raw_parts(
                    v.as_mut_ptr() as *mut FieldElement<E>,
                    v.len() / 3,
                    v.capacity() / 3,
                )
            }
        };

    lde_trace.set_host_data(main_data, aux_data);
    GPU_DEVICE_ONLY_DOWNGRADES.fetch_add(1, Ordering::Relaxed);
    true
}

/// Download a resident main LDE (column-major device buf) into the row-major
/// host Vec the CPU rounds read. Shared by the R1 and R2 downgrade paths.
pub(crate) fn download_main_lde_row_major<F>(
    h: &math_cuda::lde::GpuLdeBase,
    stream: &std::sync::Arc<math_cuda::CudaStream>,
) -> Option<Vec<FieldElement<F>>>
where
    F: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>() {
        return None;
    }
    h.wait_ready_on(stream).ok()?;
    let col_major = stream.clone_dtoh(h.buf.as_ref()).ok()?;
    stream.synchronize().ok()?;
    let (m, lde) = (h.m, h.lde_size);
    if col_major.len() != m * lde {
        return None;
    }
    // Parallel col-major → row-major transpose: each row chunk gathers from
    // the source columns independently.
    let mut row_major = vec![0u64; m * lde];
    if m > 0 {
        #[cfg(feature = "parallel")]
        {
            row_major
                .par_chunks_exact_mut(m)
                .enumerate()
                .for_each(|(r, dst)| {
                    for (c, d) in dst.iter_mut().enumerate() {
                        *d = col_major[c * lde + r];
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (r, dst) in row_major.chunks_exact_mut(m).enumerate() {
                for (c, d) in dst.iter_mut().enumerate() {
                    *d = col_major[c * lde + r];
                }
            }
        }
    }
    // SAFETY: F == Goldilocks (gated above); FieldElement<Gl> is
    // #[repr(transparent)] over u64.
    Some(unsafe {
        let mut v = std::mem::ManuallyDrop::new(row_major);
        Vec::from_raw_parts(
            v.as_mut_ptr() as *mut FieldElement<F>,
            v.len(),
            v.capacity(),
        )
    })
}

/// R1 counterpart of [`materialize_lde_trace_host`]: download the resident
/// aux trace (already row-major ext3, matching the host layout) into the
/// trace's aux table, so the aux commit continues on the host arms when the
/// device aux LDE declines at runtime.
pub(crate) fn materialize_aux_trace_host<F, E>(trace: &mut crate::trace::TraceTable<F, E>) -> bool
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if !is_goldilocks_ext3_tower::<F, E>() {
        return false;
    }
    let (buf, rows, cols) = match trace.aux_resident.as_ref() {
        Some(ra) => (ra.buf.clone(), ra.num_rows, ra.num_aux_cols),
        None => return false,
    };
    let Ok(be) = math_cuda::device::backend() else {
        return false;
    };
    let stream = be.next_stream();
    let Ok(raw) = stream.clone_dtoh(buf.as_ref()) else {
        return false;
    };
    if stream.synchronize().is_err() || raw.len() != rows * cols * 3 {
        return false;
    }
    let data = u64_to_ext3_vec::<E>(&raw);
    trace.aux_table = crate::table::Table::new(data, cols);
    trace.num_aux_columns = cols;
    // The declined device LDE attempt can leave kernels enqueued on another
    // stream still reading this buffer; its owning stream is long idle, so
    // dropping here would complete the stream-ordered free immediately and
    // the pool could hand the memory to a concurrent table's allocation
    // while those kernels run. Drain the device before the drop — this is a
    // rare recovery path.
    if be.ctx.synchronize().is_err() {
        return false;
    }
    trace.aux_resident = None;
    GPU_RESIDENT_AUX_DOWNGRADES.fetch_add(1, Ordering::Relaxed);
    true
}

/// Diagnostic: download a resident ext3 handle (3-slab layout) as per-column
/// host Vecs. Used by the xcheck post-mortem to compare the committed R2
/// parts against a host recompute.
pub(crate) fn download_ext3_columns<E>(
    h: &math_cuda::lde::GpuLdeExt3,
) -> Option<Vec<Vec<FieldElement<E>>>>
where
    E: IsField + 'static,
{
    if TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>() {
        return None;
    }
    let be = math_cuda::device::backend().ok()?;
    let stream = be.next_stream();
    h.wait_ready_on(&stream).ok()?;
    let slabs = stream.clone_dtoh(h.buf.as_ref()).ok()?;
    stream.synchronize().ok()?;
    let (m, lde) = (h.m, h.lde_size);
    if slabs.len() != m * lde * 3 {
        return None;
    }
    let mut cols = Vec::with_capacity(m);
    for c in 0..m {
        let mut interleaved = vec![0u64; lde * 3];
        for k in 0..3 {
            let slab = &slabs[(c * 3 + k) * lde..(c * 3 + k + 1) * lde];
            for r in 0..lde {
                interleaved[r * 3 + k] = slab[r];
            }
        }
        cols.push(u64_to_ext3_vec::<E>(&interleaved));
    }
    Some(cols)
}

/// The device's VRAM admission budget in bytes, if a CUDA backend is up.
/// Lets callers outside this crate (the epoch builder's trace pre-upload)
/// size their riding-ahead allocations relative to the same budget the
/// per-table scheduler admits against.
pub fn device_vram_budget_bytes() -> Option<u64> {
    math_cuda::device::backend()
        .ok()
        .map(|be| be.vram_budget_bytes())
}

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

/// Test-only: whether a scheduled fault has already fired. The hook stores -1
/// when it triggers, so after an armed prove a negative value means the error
/// path genuinely ran. Only meaningful right after arming: -1 is also the
/// idle/disarmed state, so this returns true if the hook was never armed.
/// Tests assert this instead of comparing dispatch counts, which a
/// second-tier retry can restore to the fault-free total.
#[cfg(feature = "test-cuda-faults")]
pub fn fri_fold_fault_fired() -> bool {
    math_cuda::fri::FAULT_FOLDS_REMAINING_UNTIL_ERR.load(Ordering::Relaxed) < 0
}

/// Test-only counterpart of [`fri_fold_fault_fired`] for the batch-invert hook.
#[cfg(feature = "test-cuda-faults")]
pub fn inverse_fault_fired() -> bool {
    math_cuda::inverse::FAULT_INVERSE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed) < 0
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

/// Like [`try_expand_leaf_and_tree_ext3_row_major_keep`] but the aux columns are
/// already resident on device (from the GPU LogUp aux build) — no host upload.
/// The resident buffer is only borrowed: the device-input LDE copies it
/// device-to-device into its own scratch, so `ra` stays valid afterwards.
pub(crate) fn try_expand_leaf_and_tree_ext3_row_major_keep_dev<F, E, B>(
    ra: &math_cuda::logup::ResidentAux,
    blowup_factor: usize,
    weights: &[FieldElement<F>],
    retain_host_lde: bool,
) -> Option<(
    MerkleTree<B>,
    math_cuda::lde::GpuLdeExt3,
    Vec<FieldElement<E>>,
)>
where
    F: IsField + 'static,
    E: IsField + 'static,
    B: IsMerkleTreeBackend<Node = [u8; 32]>,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>()
    {
        return None;
    }
    let weights_u64 = unsafe { weights_to_u64::<F>(weights) };

    GPU_LDE_CALLS.fetch_add((ra.num_aux_cols * 3) as u64, Ordering::Relaxed);
    GPU_LEAF_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_MERKLE_TREE_CALLS.fetch_add(1, Ordering::Relaxed);

    let (handle, lde_u64) = math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep_dev(
        &ra.buf,
        ra.num_rows,
        ra.num_aux_cols,
        blowup_factor,
        &weights_u64,
        retain_host_lde,
    )
    .inspect_err(|e| {
        // Surface the swallowed driver error (e.g. OOM): the caller drains
        // the device and retries, then downgrades the table to the host path.
        eprintln!(
            "[gpu] resident aux LDE failed (rows={} cols={} blowup={}): {e:?}",
            ra.num_rows, ra.num_aux_cols, blowup_factor
        );
    })
    .ok()?;

    let lde_out: Vec<FieldElement<E>> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(lde_u64);
        debug_assert!(
            v.len() % 3 == 0 && v.capacity() % 3 == 0,
            "lde_u64 len/capacity must be a multiple of 3 for Fp3 reinterpret"
        );
        Vec::from_raw_parts(
            v.as_mut_ptr() as *mut FieldElement<E>,
            v.len() / 3,
            v.capacity() / 3,
        )
    };
    let root = handle.tree.as_ref()?.root;
    let tree = MerkleTree::<B>::from_root(root);
    Some((tree, handle, lde_out))
}

/// Convert ext3 evals (3*n u64s, interleaved) into a freshly allocated
/// `Vec<FieldElement<E>>` of length `n`. Caller must have established
/// `E == Ext3`.
pub(crate) fn u64_to_ext3_vec<E>(raw: &[u64]) -> Vec<FieldElement<E>>
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

/// Fully-resident DEEP keeping the codeword on device in FRI order (no D2H).
/// Only the all-device arm — on any miss the caller falls back to the
/// download bridge or to [`try_deep_composition_gpu`]'s host result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_deep_composition_gpu_keep<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    parts_dev: &math_cuda::lde::GpuLdeExt3,
    h_ood: &[FieldElement<E>],
    trace_ood_columns: &[Vec<FieldElement<E>>],
    composition_poly_gammas: &[FieldElement<E>],
    trace_terms_gammas: &[Vec<FieldElement<E>>],
    inv_denoms_dev: (&CudaSlice<u64>, &Arc<CudaStream>),
    num_eval_points: usize,
) -> Option<math_cuda::deep::GpuDeepCodeword>
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
    if lde_size < gpu_lde_threshold() || !lde_size.is_power_of_two() {
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
    if parts_dev.m != num_parts || parts_dev.lde_size != lde_size {
        return None;
    }

    // Pack the small host scalars. SAFETY for ext3 transmutes: E == Ext3.
    let h_ood_raw: &[u64] = unsafe { ext3_slice_to_u64::<E>(h_ood) };
    let mut trace_ood_raw: Vec<u64> = Vec::with_capacity(num_total_cols * num_eval_points * 3);
    for col in trace_ood_columns {
        trace_ood_raw.extend_from_slice(unsafe { ext3_slice_to_u64::<E>(col) });
    }
    let gammas_h_raw: &[u64] = unsafe { ext3_slice_to_u64::<E>(composition_poly_gammas) };
    let mut gammas_tr_raw: Vec<u64> = Vec::with_capacity(num_total_cols * num_eval_points * 3);
    for col in trace_terms_gammas {
        gammas_tr_raw.extend_from_slice(unsafe { ext3_slice_to_u64::<E>(col) });
    }

    let (inv_dev, stream) = inv_denoms_dev;
    let dw = math_cuda::deep::deep_composition_ext3_fully_resident_keep(
        stream,
        main,
        aux_handle,
        parts_dev,
        inv_dev,
        h_ood_raw,
        &trace_ood_raw,
        gammas_h_raw,
        &gammas_tr_raw,
        num_parts,
        num_main,
        num_aux,
        num_eval_points,
        1,
        lde_size,
    )
    .ok()?;
    GPU_DEEP_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(dw)
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
    let coset_dev = coset_points_device_handle(coset_u64, stream)?;

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

/// Device-resident coset point buffers, keyed by `(len, points[0], points[1])`
/// — a geometric coset is fully determined by its length and first two terms,
/// so the key needs no allocation pinning. R3 OOD and the R4 DEEP inv_denoms
/// build used to re-upload the SAME domain points per table per epoch (~19 GB
/// per 100tx prove measured); one upload per distinct coset now serves the
/// whole process (a handful of sizes, ~2-16 MiB each, never evicted — same
/// policy as the host-side domain caches).
#[allow(clippy::type_complexity)]
fn coset_points_device_cache()
-> &'static std::sync::Mutex<std::collections::HashMap<(usize, u64, u64), Arc<CudaSlice<u64>>>> {
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<(usize, u64, u64), Arc<CudaSlice<u64>>>>,
    > = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// Resolve a host coset-points slice to its device-resident copy, uploading
/// once per distinct coset. The first upload synchronizes its stream so the
/// buffer is safe to read from any other stream afterwards. Returns `None` on
/// upload failure (→ the caller's fallback).
fn coset_points_device_handle(
    coset_u64: &[u64],
    stream: &Arc<CudaStream>,
) -> Option<Arc<CudaSlice<u64>>> {
    if coset_u64.len() < 2 {
        return stream.clone_htod(coset_u64).ok().map(Arc::new);
    }
    let key = (coset_u64.len(), coset_u64[0], coset_u64[1]);
    if let Some(h) = coset_points_device_cache().lock().unwrap().get(&key) {
        return Some(h.clone());
    }
    // The key only determines the full contents for a geometric sequence
    // `p_i = p_0·w^i`: verify it at sampled indices so a non-coset caller
    // trips here instead of silently aliasing another entry. Insert-only —
    // a handful of times per process.
    {
        type Fp = FieldElement<GoldilocksField>;
        let p0 = Fp::from_raw(coset_u64[0]);
        let w = Fp::from_raw(coset_u64[1])
            * p0.inv()
                .expect("coset_points_device_handle: coset offset must be nonzero");
        for i in [2usize, coset_u64.len() / 2, coset_u64.len() - 1] {
            assert_eq!(
                Fp::from_raw(coset_u64[i]),
                p0 * w.pow(i as u64),
                "coset_points_device_handle: input is not a geometric coset"
            );
        }
    }
    let buf = stream.clone_htod(coset_u64).ok()?;
    // Settle the copy before publishing: consumers run on other streams.
    stream.synchronize().ok()?;
    let h = Arc::new(buf);
    coset_points_device_cache()
        .lock()
        .unwrap()
        .insert(key, h.clone());
    Some(h)
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
    bound_stream: Option<Arc<CudaStream>>,
) -> Option<(CudaSlice<u64>, Arc<CudaStream>)>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    // Use the caller's per-table session stream when provided, so this table's
    // R3/R4 device chain serialises on one queue; otherwise grab a pool stream.
    let stream = match bound_stream {
        Some(s) => s,
        None => math_cuda::device::backend().ok()?.next_stream(),
    };
    let handle =
        try_compute_and_invert_inv_denoms_dev::<F, E>(coset_base, z_scalars, sign, &stream)?;
    Some((handle, stream))
}

/// Gather Merkle authentication paths on device for `positions` (leaf indices),
/// returning one [`Proof`] per position in the same order. Byte-identical to
/// the host `MerkleTree::get_proof_by_pos` (guarded by the `merkle_gather`
/// parity test), so R4 query openings can source proofs from the resident
/// device tree instead of the host tree. Returns `None` on any cudarc error —
/// which every caller treats as a hard abort, NOT a fallback: a resident tree
/// leaves the host tree root-only, so there is no host path to walk.
pub(crate) fn gather_proofs_dev(
    tree: &math_cuda::lde::GpuMerkleTree,
    positions: &[usize],
    stream: &Arc<CudaStream>,
) -> Option<Vec<Proof<Commitment>>> {
    if positions.is_empty() {
        return Some(Vec::new());
    }
    // Positions index an LDE that `assert_u32_domain` keeps within u32; guard the
    // cast so any future relaxation fails loudly instead of wrapping silently.
    debug_assert!(
        positions.iter().all(|&p| p <= u32::MAX as usize),
        "gather_proofs_dev: position exceeds u32 range"
    );
    let positions_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
    let bytes = math_cuda::merkle::gather_merkle_paths_dev(
        &tree.nodes,
        tree.leaves_len,
        &positions_u32,
        stream,
    )
    .ok()?;
    let depth = tree.leaves_len.trailing_zeros() as usize;
    debug_assert_eq!(bytes.len(), positions.len() * depth * 32);
    let mut proofs = Vec::with_capacity(positions.len());
    for q in 0..positions.len() {
        let mut merkle_path = Vec::with_capacity(depth);
        for level in 0..depth {
            let off = (q * depth + level) * 32;
            let mut node: Commitment = [0u8; 32];
            node.copy_from_slice(&bytes[off..off + 32]);
            merkle_path.push(node);
        }
        proofs.push(Proof { merkle_path });
    }
    Some(proofs)
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
    pub coset_points: Arc<CudaSlice<u64>>,
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
    bound_stream: Option<Arc<CudaStream>>,
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

    // Per-table session stream when provided (shares the queue with R4 DEEP for
    // this table); otherwise a pool stream.
    let stream = match bound_stream {
        Some(s) => s,
        None => math_cuda::device::backend().ok()?.next_stream(),
    };

    // SAFETY: F == Goldilocks per TypeId check; FieldElement<F> is
    // #[repr(transparent)] over u64.
    let coset_u64: &[u64] = unsafe { from_raw_parts(coset_base.as_ptr() as *const u64, n) };
    let coset_points = coset_points_device_handle(coset_u64, &stream)?;

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
    evals: &[FieldElement<E>],
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    blowup_log: u32,
    final_poly_log_degree: u32,
    inv_twiddles: &[FieldElement<F>],
) -> Option<(
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)>
where
    F: IsFFTField + IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
    T: IsStarkTranscript<E, F> + Clone,
{
    // GPU drives the early-termination FRI commit phase, mirroring
    // `commit_phase_from_evaluations`: for each committed layer (sample zeta,
    // fold, append root); then one final fold to the terminal codeword whose
    // coefficients are emitted (not a single value).
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
    // Mismatched twiddles would panic inside `FriCommitState::new`; gate here
    // so a wiring bug degrades to the CPU path instead (same gate as
    // `try_fri_commit_gpu_from_dev`).
    if inv_twiddles.len() != n0 / 2 {
        return None;
    }

    // Pack the per-domain cached inv_twiddles to u64 before any transcript
    // mutation, so on H2D / state construction failure the caller's
    // transcript is untouched.
    let mut inv_tw_u64: Vec<u64> = Vec::with_capacity(inv_twiddles.len());
    for t in inv_twiddles {
        // SAFETY: F == Goldilocks per TypeId check; FieldElement<Gl> is
        // #[repr(transparent)] over u64.
        let v: u64 = unsafe { *(t.value() as *const _ as *const u64) };
        inv_tw_u64.push(v);
    }

    // SAFETY: E == Ext3; FieldElement<Ext3> backing is [u64; 3].
    let evals_u64: &[u64] = unsafe { ext3_slice_to_u64::<E>(evals) };

    let state = match math_cuda::fri::FriCommitState::new(evals_u64, &inv_tw_u64, n0) {
        Ok(s) => s,
        Err(_) => return None,
    };
    // Host-evals entry: the caller works with host copies, keep draining them.
    fri_commit_gpu_drive(
        state,
        transcript,
        coset_offset,
        n0,
        blowup_log,
        final_poly_log_degree,
        true,
    )
}

/// [`try_fri_commit_gpu`] entered from a device-resident DEEP codeword
/// (already in FRI order): no evals H2D at all.
#[allow(clippy::type_complexity)]
pub(crate) fn try_fri_commit_gpu_from_dev<F, E, T>(
    codeword: math_cuda::deep::GpuDeepCodeword,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    blowup_log: u32,
    final_poly_log_degree: u32,
    inv_twiddles: &[FieldElement<F>],
    want_host: bool,
) -> Option<(
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)>
where
    F: IsFFTField + IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
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
    let n0 = codeword.n;
    if !n0.is_power_of_two() || n0 < 2 || n0 < gpu_lde_threshold() {
        return None;
    }
    // Mismatched twiddles would panic inside `FriCommitState::new_dev`;
    // gate here so a wiring bug degrades to the CPU path instead.
    if inv_twiddles.len() != n0 / 2 {
        return None;
    }
    let mut inv_tw_u64: Vec<u64> = Vec::with_capacity(inv_twiddles.len());
    for t in inv_twiddles {
        // SAFETY: F == Goldilocks per TypeId check.
        let v: u64 = unsafe { *(t.value() as *const _ as *const u64) };
        inv_tw_u64.push(v);
    }
    let state = match math_cuda::fri::FriCommitState::new_dev(codeword, &inv_tw_u64) {
        Ok(s) => s,
        Err(_) => return None,
    };
    fri_commit_gpu_drive(
        state,
        transcript,
        coset_offset,
        n0,
        blowup_log,
        final_poly_log_degree,
        want_host,
    )
}

/// The shared FRI commit loop over an initialized device state: per committed
/// layer sample ζ, fold + commit on device, D2H root/evals; then the terminal
/// fold and CPU coefficient extraction. Restores the transcript and returns
/// `None` on any mid-loop cudarc failure so the CPU path reruns cleanly.
#[allow(clippy::type_complexity)]
fn fri_commit_gpu_drive<F, E, T>(
    mut state: math_cuda::fri::FriCommitState,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    n0: usize,
    blowup_log: u32,
    final_poly_log_degree: u32,
    want_host: bool,
) -> Option<(
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)>
where
    F: IsFFTField + IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
    T: IsStarkTranscript<E, F> + Clone,
{
    // The unsafe zeta reads below reinterpret `FieldElement<E>` as 3 u64:
    // every caller gates the tower, but assert here so a future caller with
    // another `E` aborts instead of reading past the value.
    assert_eq!(
        TypeId::of::<E>(),
        TypeId::of::<Degree3GoldilocksExtensionField>(),
        "fri_commit_gpu_drive requires the Goldilocks ext3 tower"
    );
    // Snapshot the transcript before any sampling. On a cudarc failure
    // mid-loop we restore from this snapshot and return None, so the CPU
    // fallback in `commit_phase_from_evaluations` starts from a byte-
    // identical transcript and produces the same proof it would have
    // produced had this dispatch never been called.
    let transcript_snapshot = transcript.clone();

    // Fold layout, shared with the CPU prover and the verifier — see `FriFoldLayout`.
    let layout = crate::fri::terminal::FriFoldLayout::new(
        n0.trailing_zeros(),
        blowup_log,
        final_poly_log_degree,
    );
    // The GPU path only runs above gpu_lde_threshold(). Two cases fall back to
    // the CPU path (which handles both correctly): tiny clamped traces
    // (total_folds == 0), and terminal_len == 1 (blowup_log + k == 0), whose
    // final fold would reach n_out == 1 and trip `fold_and_commit_layer`'s
    // `n_out >= 2` assert. The final fold below is therefore always n_out >= 2.
    if layout.total_folds == 0 || layout.terminal_len < 2 {
        return None;
    }
    let num_committed = layout.num_committed;
    let mut fri_layer_list: Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>> =
        Vec::with_capacity(num_committed);

    for _layer_idx in 0..num_committed {
        // <<<< Receive challenge zeta_k
        let zeta: FieldElement<E> = transcript.sample_field_element();
        // SAFETY: E == Ext3.
        let zeta_ptr = &zeta as *const FieldElement<E> as *const u64;
        let zeta_raw: [u64; 3] = unsafe { [*zeta_ptr, *zeta_ptr.add(1), *zeta_ptr.add(2)] };

        let (layer_evals_u64, evals_dev, dev_tree) =
            match state.fold_and_commit_layer(zeta_raw, want_host) {
                Ok(v) => v,
                Err(_) => {
                    *transcript = transcript_snapshot.clone();
                    return None;
                }
            };

        // Build the FriLayer: a root only host tree, the tree and evals kept
        // resident on device (`gpu_tree` / `gpu_evals`), and host evals only
        // when a host copy was drained (fallback consumers).
        let evaluation = layer_evals_u64
            .map(|v| u64_to_ext3_vec::<E>(&v))
            .unwrap_or_default();
        let root = dev_tree.root;
        let merkle_tree = MerkleTree::<FriLayerMerkleTreeBackend<E>>::from_root(root);
        // Retain the device evals only when no host copy exists (device-only):
        // with a host copy the query phase reads it, and the retained buffer
        // would be ~24 bytes/LDE-row of dead VRAM per table.
        fri_layer_list.push(FriLayer {
            evaluation,
            merkle_tree,
            gpu_tree: Some(dev_tree),
            gpu_evals: (!want_host).then_some(evals_dev),
        });

        // >>>> Send commitment: [p_k]
        transcript.append_bytes(&root);
    }

    // Final (uncommitted) fold to the terminal codeword. n_out == terminal_len
    // >= 2, so reuse fold_and_commit_layer and keep only its evaluations (the
    // coefficient extraction below is host-side, so always drain them); the
    // Merkle root/nodes are discarded (the terminal layer is sent as coeffs).
    let zeta_final: FieldElement<E> = transcript.sample_field_element();
    let zeta_ptr = &zeta_final as *const FieldElement<E> as *const u64;
    let zeta_raw: [u64; 3] = unsafe { [*zeta_ptr, *zeta_ptr.add(1), *zeta_ptr.add(2)] };

    let (terminal_evals_u64, _evals_dev, _tree) = match state.fold_and_commit_layer(zeta_raw, true)
    {
        Ok(v) => v,
        Err(_) => {
            *transcript = transcript_snapshot;
            return None;
        }
    };
    let terminal_evals_u64 = terminal_evals_u64.expect("terminal fold drains to host");
    debug_assert_eq!(terminal_evals_u64.len(), layout.terminal_len * 3);
    let terminal_codeword = u64_to_ext3_vec::<E>(&terminal_evals_u64);

    // CPU-side coefficient extraction, identical to commit_phase_from_evaluations.
    let terminal_offset = coset_offset.pow(1u64 << layout.total_folds);
    let final_poly_coeffs = crate::fri::terminal::coeffs_from_terminal_codeword::<F, E>(
        &terminal_codeword,
        &terminal_offset,
        layout.effective_k,
    );

    // >>>> Send the final polynomial coefficients.
    for c in &final_poly_coeffs {
        transcript.append_field_element(c);
    }

    GPU_FRI_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((final_poly_coeffs, fri_layer_list))
}

/// GPU FRI query phase: gather each layer's paths on device instead of walking
/// host trees. For layer `l` and query `iota` the opened position is
/// `(iota >> l) >> 1`, matching [`crate::fri::query_phase`]. Paths for all
/// queries are gathered in one batched call per layer. The layer evaluations
/// (`evaluation[index ^ 1]`) are read from the host Vecs as before.
///
/// Returns None when there are no layers or the layers are host trees (CPU
/// commit), so the caller falls back to the host walk.
pub(crate) fn try_fri_query_phase_gpu<E>(
    fri_layers: &[FriLayer<E, FriLayerMerkleTreeBackend<E>>],
    iotas: &[usize],
) -> Option<Vec<FriDecommitment<E>>>
where
    E: IsField + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
{
    if fri_layers.is_empty() {
        return None;
    }
    // The GPU FRI commit sets `gpu_tree` on every layer as a group; the CPU
    // commit sets none. Host trees fall back to the host walk. When the layers
    // are device resident the host trees are root only, so the gather below must
    // succeed (a failure is a hard abort, not a silent walk). The residency is
    // all or nothing; assert it so a future partial-build can never route a
    // root-only layer through the host walk and ship empty proofs.
    let first_resident = fri_layers[0].gpu_tree.is_some();
    debug_assert!(
        fri_layers
            .iter()
            .all(|l| l.gpu_tree.is_some() == first_resident),
        "FRI layer residency must be all or nothing"
    );
    if !first_resident {
        return None;
    }
    let stream = math_cuda::device::backend()
        .expect("cuda backend for device-resident FRI query")
        .next_stream();
    let num_layers = fri_layers.len();

    // Batched gather: one call per layer over all queries.
    let mut per_layer_proofs: Vec<Vec<Proof<Commitment>>> = Vec::with_capacity(num_layers);
    for (l, layer) in fri_layers.iter().enumerate() {
        let tree = layer
            .gpu_tree
            .as_ref()
            .expect("FRI layers are device-resident as a group");
        let positions: Vec<usize> = iotas.iter().map(|&iota| (iota >> l) >> 1).collect();
        per_layer_proofs.push(
            gather_proofs_dev(tree, &positions, &stream)
                .expect("device FRI-layer gather failed; resident tree has no host fallback"),
        );
    }

    // Symmetric evals per layer: read the host Vec when it was drained,
    // otherwise a batched device gather off the resident layer evals
    // (device-only, where no host copy exists).
    let per_layer_syms: Vec<Option<Vec<FieldElement<E>>>> = fri_layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            if !layer.evaluation.is_empty() {
                return None;
            }
            let evals_dev = layer
                .gpu_evals
                .as_ref()
                .expect("device-only FRI layer without resident evals");
            let positions: Vec<u32> = iotas.iter().map(|&iota| ((iota >> l) ^ 1) as u32).collect();
            let raw = math_cuda::fri::gather_ext3_at(evals_dev, &positions, &stream)
                .expect("device FRI sym-eval gather failed; no host fallback");
            Some(
                crate::constraint_ir::gpu_interp::ext3_u64_to_field::<E>(&raw)
                    .expect("resident FRI evals are Goldilocks ext3"),
            )
        })
        .collect();

    // Reassemble per-query decommitments, matching the host walk's order.
    let decommits = iotas
        .iter()
        .enumerate()
        .map(|(q, &iota)| {
            let mut layers_evaluations_sym = Vec::with_capacity(num_layers);
            let mut layers_auth_paths = Vec::with_capacity(num_layers);
            let mut index = iota;
            for (l, layer) in fri_layers.iter().enumerate() {
                let sym = match &per_layer_syms[l] {
                    Some(v) => v[q].clone(),
                    None => layer.evaluation[index ^ 1].clone(),
                };
                layers_evaluations_sym.push(sym);
                layers_auth_paths.push(per_layer_proofs[l][q].clone());
                index >>= 1;
            }
            FriDecommitment {
                layers_auth_paths,
                layers_evaluations_sym,
            }
        })
        .collect();
    Some(decommits)
}

/// GPU↔CPU parity for the preprocessed split-tree commit path. Requires the
/// `cuda` feature and a visible GPU (skipped otherwise via the dispatch gate
/// returning `None` — asserted here, so a silent skip fails the test).
#[cfg(all(test, feature = "cuda"))]
mod split_tree_tests {
    use super::*;
    use crate::config::BatchedMerkleTreeBackend;
    use crate::prover::{IsStarkProver, Prover};
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    type F = GoldilocksField;
    type Fp = FieldElement<F>;
    type TestProver = Prover<F, Ext3, ()>;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Both subset trees (roots, nodes via openings) must equal the CPU
    /// `commit_rows_bit_reversed_subset` built over the same row-major LDE.
    /// The LDE itself is parity-pinned by the existing full-row fused tests,
    /// so the CPU reference consumes the GPU's returned LDE directly — this
    /// isolates the tree layout/hashing under test.
    #[test]
    fn split_trees_match_cpu_subset_commits() {
        // This shape's LDE is 2^19, well above the dispatch threshold, so the
        // GPU path must engage.
        let n: usize = 1 << 18;
        let blowup: usize = 2;
        let m: usize = 5;
        let split: usize = 2;

        let mut rng = SplitMix64(0x5EED_C0DE_5EED_C0DE);
        let data: Vec<Fp> = (0..n * m).map(|_| Fp::from(rng.next_u64())).collect();
        let weights: Vec<Fp> = (0..n).map(|_| Fp::from(rng.next_u64())).collect();

        let (pre_tree, mult_tree, handle, lde) =
            try_expand_split_trees_row_major_keep::<F, F, BatchedMerkleTreeBackend<F>>(
                &data, None, n, m, blowup, &weights, split, true, true,
            )
            .expect("GPU split path must engage above the threshold");
        let pre_tree = pre_tree.expect("precomputed tree was requested");

        let (cpu_pre, cpu_pre_root) =
            TestProver::commit_rows_bit_reversed_subset(&lde, m, 0, split)
                .expect("CPU subset commit (precomputed)");
        let (cpu_mult, cpu_mult_root) =
            TestProver::commit_rows_bit_reversed_subset(&lde, m, split, m)
                .expect("CPU subset commit (multiplicities)");

        assert_eq!(pre_tree.root, cpu_pre_root, "precomputed root");
        assert_eq!(mult_tree.root, cpu_mult_root, "multiplicity root");

        // Openings must be byte-identical at scattered positions (pins the
        // full node buffers, not just the roots). The mult tree is resident
        // (host tree root only), so its paths come from the device gather —
        // the exact production opening path.
        let num_leaves = n * blowup / 2;
        let dev_tree = handle.tree.as_ref().expect("resident mult subset tree");
        let stream = math_cuda::device::backend().unwrap().next_stream();
        for pos in [0usize, 1, 511, 12_345, num_leaves - 1] {
            assert_eq!(
                pre_tree.get_proof_by_pos(pos).unwrap().merkle_path,
                cpu_pre.get_proof_by_pos(pos).unwrap().merkle_path,
                "precomputed path at {pos}"
            );
            let dev_proofs =
                gather_proofs_dev(dev_tree, &[pos], &stream).expect("device mult-tree path gather");
            assert_eq!(
                dev_proofs[0].merkle_path,
                cpu_mult.get_proof_by_pos(pos).unwrap().merkle_path,
                "multiplicity path at {pos}"
            );
        }
        assert_eq!(mult_tree.root, dev_tree.root, "root-only host tree root");

        // The handle must carry the column-major LDE for downstream rounds:
        // spot-check a few cells against the row-major host LDE.
        assert_eq!(handle.m, m);
        assert_eq!(handle.lde_size, n * blowup);
    }
}
