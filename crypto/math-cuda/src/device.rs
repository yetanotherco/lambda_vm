//! CUDA device context, stream pool, kernel handles, and twiddle cache.
//!
//! One process-wide backend — lazy-initialised on first use. All kernels live
//! on a single CUDA context; a pool of streams lets rayon-parallel callers
//! overlap H2D / compute / D2H.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream};
use cudarc::nvrtc::Ptx;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;

use crate::Result;
use crate::ntt::{twiddles_forward, twiddles_inverse};

/// Reusable pinned host staging buffer. Shared across all streams via a
/// `Mutex` (see `Backend::pinned_staging`); the LDE call holds the lock
/// across the D2H + memcpy-to-user-Vecs window.
///
/// Allocated with `cuMemHostAlloc(flags=0)` — portable, non-write-combined,
/// so both DMA writes from device and CPU reads into user Vecs run at full
/// speed. Grows power-of-two; never shrinks.
pub struct PinnedStaging {
    ptr: *mut u64,
    capacity_elems: usize,
    /// Reusable completion event for [`async_dtoh_via`] copies through this
    /// slot. Created once on first use and re-recorded per drain — per-call
    /// cuEventCreate/Destroy measurably convoys the driver lock under load.
    /// At most one drain per slot is in flight (the pending holds the slot
    /// mutex), so a single event can never be aliased.
    event: Option<cudarc::driver::CudaEvent>,
}

// SAFETY: the raw pointer aliases host memory allocated via cuMemHostAlloc.
// We guard concurrent access with a Mutex; the pointer is valid for the
// lifetime of this struct and is freed on drop.
unsafe impl Send for PinnedStaging {}
unsafe impl Sync for PinnedStaging {}

impl PinnedStaging {
    fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            capacity_elems: 0,
            event: None,
        }
    }

    pub fn ensure_capacity(&mut self, min_elems: usize, ctx: &CudaContext) -> Result<()> {
        if self.capacity_elems >= min_elems {
            return Ok(());
        }
        // cuMemHostAlloc requires the context to be current on this thread.
        ctx.bind_to_thread()?;
        // Free old (if any) before allocating the new one.
        if !self.ptr.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut _);
            }
            self.ptr = std::ptr::null_mut();
            self.capacity_elems = 0;
        }
        let new_cap = min_elems.next_power_of_two().max(1 << 20); // at least 8 MB
        let bytes = new_cap * std::mem::size_of::<u64>();
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(bytes, 0 /* flags: non-WC */)?
        } as *mut u64;
        self.ptr = ptr;
        self.capacity_elems = new_cap;
        Ok(())
    }

    /// Record the slot's reusable event on `stream` (creating it on first use;
    /// normally pre-created at backend init so no mid-prove cuEventCreate).
    /// Pairs with [`PinnedStaging::sync_event`]; also re-recorded by
    /// [`async_dtoh_via`] — safe because slot access is serialized by its
    /// mutex, so a recorded event is always synchronized before re-recording.
    pub fn record_event(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        match self.event.as_ref() {
            Some(ev) => ev.record(stream),
            None => {
                self.event = Some(stream.record_event(None)?);
                Ok(())
            }
        }
    }

    /// Block until the last [`PinnedStaging::record_event`] point completes.
    pub fn sync_event(&self) -> Result<()> {
        match self.event.as_ref() {
            Some(ev) => ev.synchronize(),
            None => Ok(()),
        }
    }

    /// View of the first `len` elements. Caller must hold this `PinnedStaging`
    /// locked while using the slice; the slice aliases the internal pointer.
    ///
    /// # Safety
    /// Caller must not outlive the `PinnedStaging` and must not race with
    /// concurrent uses.
    pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [u64] {
        assert!(len <= self.capacity_elems);
        if len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl Drop for PinnedStaging {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut _);
            }
        }
    }
}

// Kernels are AOT-compiled to native cubin (SASS) by build.rs, embedded here,
// and loaded via `Ptx::from_binary` (cubin bytes -> cuModuleLoadData). This
// avoids the PTX-ISA/driver-version JIT check — see build.rs `compile_kernel`.
// An empty slice (nvcc-less stub build) fails to load at runtime and the caller
// falls back to CPU.
const ARITH_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/arith.cubin"));
const NTT_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ntt.cubin"));
const KECCAK_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/keccak.cubin"));
const RPX_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rpx.cubin"));
const BARY_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/barycentric.cubin"));
const DEEP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deep.cubin"));
const FRI_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fri.cubin"));
const INVERSE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inverse.cubin"));
const LOGUP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/logup.cubin"));
const CONSTRAINT_INTERP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/constraint_interp.cubin"));
const BLAKE3_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/blake3.cubin"));
const SUMCHECK_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sumcheck.cubin"));
const WHIR_FOLD_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/whir_fold.cubin"));

/// Number of CUDA streams in the pool. Larger pools let many rayon-parallel
/// callers overlap on the GPU without serializing on stream ownership. The
/// default stream is deliberately excluded because it synchronises with all
/// other streams, defeating the point of the pool.
const STREAM_POOL_SIZE: usize = 32;

pub struct Backend {
    pub ctx: Arc<CudaContext>,
    streams: Vec<Arc<CudaStream>>,
    /// Per-rayon-worker pinned staging buffers. Indexed by
    /// `rayon::current_thread_index()` (0 for non-rayon callers). Each slot
    /// grows lazily on first use, idle slots stay at zero allocation.
    /// Worst-case footprint is `N_workers × max_LDE_size` of pinned host RAM.
    pinned_staging: Vec<Mutex<PinnedStaging>>,
    /// Per-worker pinned staging for Merkle leaf hashes. Same layout as
    /// `pinned_staging`; sized `num_rows * 32` bytes per slot. Lives
    /// alongside the LDE staging so the GPU→host D2H runs at PCIe line-rate.
    pinned_hashes: Vec<Mutex<PinnedStaging>>,
    util_stream: Arc<CudaStream>,
    /// Free-list of pre-created events for [`Backend::take_event`].
    event_pool: Mutex<Vec<cudarc::driver::CudaEvent>>,
    next: AtomicUsize,
    /// VRAM budget (bytes) for admission control. See
    /// [`detect_vram_budget_bytes`].
    vram_budget_bytes: u64,
    /// Device bytes promised to structures that are still alive. See
    /// [`Backend::reserve`].
    reserved: AtomicU64,

    // arith.cubin
    pub vector_add_u64: CudaFunction,
    pub gl_add: CudaFunction,
    pub gl_sub: CudaFunction,
    pub gl_mul: CudaFunction,
    pub gl_neg: CudaFunction,
    pub ext3_mul: CudaFunction,
    pub ext3_add: CudaFunction,
    pub ext3_sub: CudaFunction,

    // ntt.cubin
    pub bit_reverse_permute: CudaFunction,
    pub lift_spread: CudaFunction,
    pub mobius_level: CudaFunction,
    pub mobius_low_levels: CudaFunction,
    pub mobius_tile: CudaFunction,
    pub ntt_dit_level: CudaFunction,
    pub ntt_dit_8_levels: CudaFunction,
    pub ntt_dit_tile: CudaFunction,
    pub pointwise_mul: CudaFunction,
    pub scalar_mul: CudaFunction,
    pub bit_reverse_permute_batched: CudaFunction,
    pub ntt_dit_level_batched: CudaFunction,
    pub ntt_dit_8_levels_batched: CudaFunction,
    pub pointwise_mul_batched: CudaFunction,
    pub scalar_mul_batched: CudaFunction,
    // row-major NTT kernels
    pub bit_reverse_row_major: CudaFunction,
    pub ntt_dit_level_row_major: CudaFunction,
    pub ntt_dit_8_levels_row_major: CudaFunction,
    pub pointwise_mul_row_major: CudaFunction,
    pub matrix_transpose_strided: CudaFunction,

    // keccak.cubin
    pub keccak256_leaves_base_row_major_row_pair: CudaFunction,
    pub keccak256_leaves_base_row_major_row_pair_range: CudaFunction,
    pub keccak256_leaves_base_batched: CudaFunction,
    pub keccak256_leaves_base_coset: CudaFunction,
    pub keccak256_leaves_ext3_coset: CudaFunction,
    pub keccak256_leaves_base_row_pair_batched: CudaFunction,
    pub keccak256_leaves_ext3_batched: CudaFunction,
    pub grind_search: CudaFunction,
    pub keccak_comp_poly_leaves_ext3: CudaFunction,
    pub keccak_fri_leaves_ext3: CudaFunction,
    pub keccak_merkle_level: CudaFunction,
    pub keccak_merkle_tail: CudaFunction,
    pub merkle_gather_paths: CudaFunction,
    // blake3.cubin — the leaf kernels, the Merkle level/tail compressors, and
    // the parity-harness probes that are the only host-visible handle on the
    // device compression function, byte serialization and chain construction
    // (see `kernels/blake3.cu`). Twin for twin with the keccak set above, and in
    // the same order. `merkle_gather_paths` has no twin: path gathering copies
    // nodes and never hashes, so it is hash-agnostic and both trees share it.
    //
    // Keccak stays the prover's default, so no production dispatch reaches these
    // yet — they exist so the GPU can follow the CPU's hash switch (PA-PLAN §6.1).
    pub blake3_leaves_base_row_major_row_pair: CudaFunction,
    pub blake3_leaves_base_row_major_row_pair_range: CudaFunction,
    pub blake3_leaves_base_batched: CudaFunction,
    pub blake3_leaves_base_row_pair_batched: CudaFunction,
    pub blake3_leaves_ext3_batched: CudaFunction,
    pub blake3_comp_poly_leaves_ext3: CudaFunction,
    pub blake3_fri_leaves_ext3: CudaFunction,
    pub blake3_merkle_level: CudaFunction,
    pub blake3_merkle_tail: CudaFunction,
    pub blake3_compress_probe_6r: CudaFunction,
    pub blake3_compress_probe_7r: CudaFunction,
    pub blake3_compress_probe_default: CudaFunction,
    pub blake3_rounds_probe: CudaFunction,
    pub blake3_serialize_felts_probe: CudaFunction,
    pub blake3_blocks_of_felts_probe: CudaFunction,
    pub blake3_chain_probe: CudaFunction,

    // rpx.cubin — the RPX256 (XHash12) leaf kernels, Merkle level/tail
    // compressors and the permutation probe (see `kernels/rpx.cu`). Twin for
    // twin with the blake3 set above and in the same order; the probe is the
    // only host-visible handle on the bare device permutation, which the parity
    // tests check against the host `Rpx256`.
    pub rpx_leaves_base_row_major_row_pair: CudaFunction,
    pub rpx_leaves_base_row_major_row_pair_range: CudaFunction,
    pub rpx_leaves_base_batched: CudaFunction,
    pub rpx_leaves_base_row_pair_batched: CudaFunction,
    pub rpx_leaves_ext3_batched: CudaFunction,
    pub rpx_comp_poly_leaves_ext3: CudaFunction,
    pub rpx_fri_leaves_ext3: CudaFunction,
    pub rpx_merkle_level: CudaFunction,
    pub rpx_merkle_tail: CudaFunction,
    pub rpx_permute_probe: CudaFunction,
    pub rpx_grind_search: CudaFunction,
    /// ⛔ DIAGNOSTIC ONLY — the grind search with its executed-permutation
    /// counters. Nothing on a proving path launches it; its one caller is
    /// [`crate::grinding::search_counted`], which reads whether a slow launch
    /// does MORE work or the same work more slowly.
    pub rpx_grind_search_counted: CudaFunction,

    // rpx.cubin — the algebraic hash's twins of the keccak entries above.
    // Only the ones the WHIR path reaches are bound: the coset leaves, the two
    // tree compressors and the grind. The row-group leaf kernels the per-table
    // branch uses are in the cubin but are not loaded here, because nothing on
    // this path launches them and an unused handle is a claim that something
    // does.
    pub rpx_leaves_base_coset: CudaFunction,
    pub rpx_leaves_ext3_coset: CudaFunction,

    // barycentric.cubin
    pub barycentric_base_batched: CudaFunction,
    pub barycentric_ext3_batched: CudaFunction,
    pub barycentric_base_batched_strided: CudaFunction,
    pub barycentric_ext3_batched_strided: CudaFunction,
    pub barycentric_base_strided_multi: CudaFunction,
    pub barycentric_ext3_strided_multi: CudaFunction,
    pub barycentric_combine_partials: CudaFunction,
    pub gather_rows_base: CudaFunction,
    pub gather_rows_ext3: CudaFunction,

    // deep.cubin
    pub deep_composition_ext3_row: CudaFunction,
    pub bit_reverse_ext3_kernel: CudaFunction,

    // fri.cubin
    pub fri_fold_ext3: CudaFunction,
    pub gather_ext3_at: CudaFunction,
    pub fri_update_twiddles: CudaFunction,

    // inverse.cubin
    pub compute_denoms_ext3: CudaFunction,
    pub block_inclusive_scan_fwd_ext3: CudaFunction,
    pub apply_block_offsets_fwd_ext3: CudaFunction,
    pub block_inclusive_scan_rev_ext3: CudaFunction,
    pub apply_block_offsets_rev_ext3: CudaFunction,
    pub batch_inverse_combine_ext3: CudaFunction,
    pub invert_total_ext3: CudaFunction,
    pub logup_fingerprint_ext3: CudaFunction,
    pub logup_term_ext3: CudaFunction,
    pub logup_row_sum_ext3: CudaFunction,
    pub logup_scan_block_add_ext3: CudaFunction,
    pub logup_apply_offsets_add_ext3: CudaFunction,
    pub logup_finalize_accum_ext3: CudaFunction,
    pub logup_assemble_aux_ext3: CudaFunction,

    // sumcheck.cubin
    pub sumcheck_round_ext3: CudaFunction,
    pub sum_partials_ext3: CudaFunction,
    pub sumcheck_fold_ext3: CudaFunction,
    pub mle_fold_base_ext3: CudaFunction,
    pub eq_expand_level_ext3: CudaFunction,
    pub eq_seed_shares_ext3: CudaFunction,
    pub eq_expand_level_shares_ext3: CudaFunction,
    pub program_map_ext3: CudaFunction,
    pub factors_from_columns_ext3: CudaFunction,
    pub mle_lift_base_ext3: CudaFunction,
    pub add_scaled_ext3: CudaFunction,
    pub fill_ext3: CudaFunction,
    pub fraction_fold_ext3: CudaFunction,
    pub fraction_fold_padded_ext3: CudaFunction,
    pub mle_fold_base_ext3_many: CudaFunction,

    // whir_fold.cubin
    pub whir_fold_base_ext3: CudaFunction,
    pub whir_fold_ext3: CudaFunction,
    pub gather_cosets: CudaFunction,

    // constraint_interp.cubin
    pub constraint_interp_kernel: CudaFunction,
    pub constraint_composition_kernel: CudaFunction,
    pub decompose_d2_kernel: CudaFunction,
    pub comp_h_to_slabs_kernel: CudaFunction,

    // Twiddle caches keyed by log_n.
    fwd_twiddles: Mutex<Vec<Option<Arc<CudaSlice<u64>>>>>,
    inv_twiddles: Mutex<Vec<Option<Arc<CudaSlice<u64>>>>>,
}

/// The environment knob for the device default memory pool's release
/// threshold, in MiB: the bytes of freed stream-ordered memory the pool keeps
/// before handing memory back to the OS at the next sync. Unset means
/// [`DEFAULT_MEMPOOL_RELEASE_THRESHOLD_BYTES`]. The VRAM sampler runs set it
/// to `0`, so `total - free` reads the live working set and not the retained
/// pool.
pub const MEMPOOL_RELEASE_ENV: &str = "LAMBDA_VM_MEMPOOL_RELEASE_MB";

/// Retain every freed block (`u64::MAX`): a same-shape allocation skips the
/// driver and reuses the block, which is what the per-table pipeline's
/// repeated LDE/FRI buffers want.
///
/// Measured, not guessed (RTX 5090, 2026-09-07, `one_lde_buffer::vram_arm`
/// at 2^21 × 316 @ blowup 2, five commits, in-process 1 kHz peak): retain-all
/// 1176.9 / 1057.9 / 1055.3 / 1056.2 / 1055.6 ms against release-0
/// 1178.5 / 1057.2 / 1077.5 / 1081.2 / 1079.3 ms — retention ≈2% faster once
/// the first commit has populated the pool — and a peak of 15.67 GiB under
/// both: a same-shape allocation reuses the retained block, so retention adds
/// nothing to the peak. The unequal-shape case is covered at block scale by
/// the multi-table q=41 wrap rung under this default (VRAM peak 28,976 MiB, no
/// device decline): the stream-ordered allocator serves a new request from the
/// physical chunks it retains, and the release threshold governs only what a
/// sync hands back to the OS. The explicit release for a moment reuse cannot
/// serve is [`Backend::trim_mempool_to`]; the sampler runs set the knob to `0`
/// so `total - free` reads the live set rather than the pool.
pub const DEFAULT_MEMPOOL_RELEASE_THRESHOLD_BYTES: u64 = u64::MAX;

/// The effective release threshold in bytes: the knob when set and parseable,
/// the default otherwise. Read once per process; the prover's diagnostics
/// print it so every box log states the posture its run had.
pub fn mempool_release_threshold_bytes() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var(MEMPOOL_RELEASE_ENV)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(DEFAULT_MEMPOOL_RELEASE_THRESHOLD_BYTES)
    })
}

/// The device default memory pool, or `None` on a device/driver without
/// stream-ordered allocator support.
///
/// # Safety
///
/// `ctx` must be a live context; its device is queried directly.
unsafe fn default_mempool(ctx: &CudaContext) -> Option<cudarc::driver::sys::CUmemoryPool> {
    use cudarc::driver::sys;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: the out-pointer is a valid stack slot; the device is the
    // context's own.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool as *mut _, ctx.cu_device())
            .result()
            .ok()
            .map(|()| pool)
    }
}

/// Set the device default memory pool's release threshold
/// ([`mempool_release_threshold_bytes`]) so freed stream-ordered allocations
/// are kept for reuse instead of returned to the OS at each sync. Best-effort:
/// any failure leaves the driver default (release everything) untouched, and
/// the one-line report says so.
fn retain_default_mempool(ctx: &CudaContext) {
    use cudarc::driver::sys;
    let threshold = mempool_release_threshold_bytes();
    // SAFETY: raw CUDA driver calls on the just-created context's device; the
    // threshold is read as a u64 by the driver. Errors are swallowed.
    let set = unsafe {
        default_mempool(ctx).is_some_and(|pool| {
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &threshold as *const u64 as *mut core::ffi::c_void,
            )
            .result()
            .is_ok()
        })
    };
    // One line per process, so the box log states the posture the run had.
    eprintln!(
        "[gpu] mempool release threshold: {}{}",
        match threshold {
            u64::MAX => "retain all freed blocks".to_string(),
            t => format!("{} MiB", t >> 20),
        },
        if set {
            ""
        } else {
            " (driver refused; the release-on-sync default stays)"
        }
    );
}

/// Device bytes held for as long as this lives. See [`Backend::reserve`].
///
/// ★ The count is atomic and the reservation is GROWABLE, because a chain
/// promises its room once and then discovers more of it: the tree a commitment
/// caches is not known when the codeword reserves, and it is shared through an
/// `Arc` by the time it is. Growing this rather than taking a second
/// reservation is what keeps ONE number answering "what does this chain hold" —
/// two accountings for one working set is a shape this codebase has paid for
/// before.
#[derive(Debug)]
pub struct DeviceReservation {
    bytes: AtomicU64,
}

impl DeviceReservation {
    /// Bytes this reservation currently accounts for.
    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Promise `extra` more against the same budget, under this reservation.
    ///
    /// Returns false and changes nothing if the budget will not take it — the
    /// caller then does without whatever it wanted the bytes for, rather than
    /// holding memory the accounting cannot see.
    pub fn grow(&self, extra: u64) -> bool {
        let Ok(be) = backend() else { return false };
        let mut held = be.reserved.load(Ordering::Relaxed);
        loop {
            if held.saturating_add(extra) > be.vram_budget_bytes {
                return false;
            }
            match be.reserved.compare_exchange_weak(
                held,
                held + extra,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.bytes.fetch_add(extra, Ordering::Relaxed);
                    return true;
                }
                Err(seen) => held = seen,
            }
        }
    }

    /// Give `given` of them back, when what they were promised for is dropped
    /// before the reservation is.
    pub fn shrink(&self, given: u64) {
        let given = given.min(self.bytes.load(Ordering::Relaxed));
        if given == 0 {
            return;
        }
        self.bytes.fetch_sub(given, Ordering::Relaxed);
        if let Ok(be) = backend() {
            be.reserved.fetch_sub(given, Ordering::Relaxed);
        }
    }
}

impl Drop for DeviceReservation {
    fn drop(&mut self) {
        if let Ok(be) = backend() {
            be.reserved
                .fetch_sub(self.bytes.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }
}

/// Hands the device default memory pool's retained blocks back to the OS.
///
/// The pool keeps freed stream-ordered allocations forever by design (see
/// [`retain_default_mempool`]), which is what makes repeated allocations cheap
/// — and also what makes a *new* shape of allocation fail while the pool sits
/// on memory it is not using. Best-effort: a failure leaves things as they are.
fn trim_default_mempool() {
    use cudarc::driver::sys;
    let Ok(be) = backend() else { return };
    // SAFETY: raw driver calls. The device is the backend's, the out-pointer a
    // stack slot, and the target size is read as a u64. Errors are swallowed.
    unsafe {
        let dev = be.ctx.cu_device();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool as *mut _, dev)
            .result()
            .is_err()
        {
            return;
        }
        let _ = sys::cuMemPoolTrimTo(pool, 0).result();
    }
}

/// Lands every stream's pending frees and hands the pool's retained blocks
/// back.
///
/// The frees are stream-ordered, so a buffer dropped on another stream is not
/// free until that stream reaches the drop — and the pool cannot return what
/// it has not been given yet. Draining the whole context first is what makes
/// the trim worth doing.
///
/// Public because a test that asks the driver how much memory this process has
/// taken needs the pool empty first, or it is measuring the pool's retention
/// instead of the caller's — see `a_group_holds_only_its_codewords_before_any_open`.
pub fn drain_and_trim() -> Result<()> {
    let be = backend()?;
    be.ctx.synchronize()?;
    trim_default_mempool();
    Ok(())
}

/// Promises `bytes` against the budget for a caller whose structure outlives
/// the type that spends them.
pub fn reserve(bytes: u64) -> Option<DeviceReservation> {
    backend().ok()?.reserve(bytes)
}

/// Argue-surface device fallbacks: the reservation refusals in math-cuda's
/// `sumcheck`, `gkr` and `columns`, counted where each one's `reserve` returns
/// `None` and its work moves to the host.
///
/// ⛔ WHY THIS EXISTS. Until this counter the only fallback number the campaign
/// read was `multilinear::gpu::host_fallbacks()`, which has ONE caller — the
/// COMMIT path (`multilinear/src/whir_chain.rs`) — so every `host fallbacks 0`
/// certified that no COMMITMENT fell back and said NOTHING about the per-table
/// ARGUMENT. wt16 was net-negative for exactly that blind spot: the leaf-layer
/// retention grew `be.reserved`, argue's `reserve` then refused and moved to
/// the host UNCOUNTED, and the slot-level reading looked like a clean win.
/// Read beside `host_fallbacks()`, this makes "the device did the work"
/// distinguishable from "it quietly did not" on the argue surface.
///
/// SCOPE, stated precisely. The FIVE argue-side `reserve`→`None` sites in
/// `crypto/math-cuda/src`: `sumcheck.rs` (×3), `gkr.rs` (×1), `columns.rs`
/// (×1). This is NOT the whole device surface, and it does not claim to be:
/// `multilinear/src/gpu.rs` holds two further argue-side sites — `:177`
/// (`reserve_room`) and `:1572` (the GKR tree) — whose `None` still falls to
/// the host uncounted. Those are a documented FOLLOW-UP, out of this counter's
/// scope, because each needs its caller traced before it can honestly be
/// labelled a fallback. A THIRD site there, `:1551`, is a SPECULATIVE reserve
/// whose `None` selects a lazy path that is STILL on the device — NOT a
/// fallback, and it must never be counted. Putting a wrong site into the very
/// counter meant to end false numbers is the one thing to avoid.
static DEVICE_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Argue-surface device fallbacks this process has taken — see
/// [`DEVICE_FALLBACKS`] for the enumerated sites and the scope it does not
/// cover. Read alongside `multilinear::gpu::host_fallbacks()` (the commit-side
/// count) for both surfaces.
pub fn device_fallbacks() -> u64 {
    DEVICE_FALLBACKS.load(Ordering::Relaxed)
}

/// Zero the process-wide counter. For a test that wants to assert a delta, and
/// for a harness that reads one prove's worth from a reused process.
pub fn reset_device_fallbacks() {
    DEVICE_FALLBACKS.store(0, Ordering::Relaxed);
}

/// Record one argue-side reservation refusal — bumped at each of the five
/// sites [`DEVICE_FALLBACKS`] enumerates, and nowhere else.
pub(crate) fn note_device_fallback() {
    DEVICE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// Allocates on `stream`, and if the device says no, gives the pool's retained
/// blocks back and asks once more.
///
/// "Out of memory" from the stream-ordered allocator usually means the pool is
/// holding blocks of the wrong shape, not that the device is full — a second
/// prover on the same card is enough. It is worth one retry: a prove whose
/// tables are already on the device has nowhere to fall back to, so a failure
/// here is a failed proof.
///
/// # Safety
/// The caller must write every element before reading it, as with
/// `CudaStream::alloc`.
pub unsafe fn alloc_or_trim<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    // SAFETY: the caller's, forwarded.
    match unsafe { stream.alloc::<T>(len) } {
        Ok(slice) => Ok(slice),
        Err(_) => {
            drain_and_trim()?;
            // SAFETY: the caller's, forwarded.
            unsafe { stream.alloc::<T>(len) }
        }
    }
}

/// Uploads on `stream`, with the same one retry as [`alloc_or_trim`]: a copy
/// to the device allocates too, and the small ones are no less fatal for being
/// small.
pub fn htod_or_trim<T: cudarc::driver::DeviceRepr + Unpin>(
    stream: &Arc<CudaStream>,
    src: &[T],
) -> Result<CudaSlice<T>> {
    match stream.clone_htod(src) {
        Ok(slice) => Ok(slice),
        Err(_) => {
            drain_and_trim()?;
            stream.clone_htod(src)
        }
    }
}

/// The same, zeroed.
pub fn alloc_zeros_or_trim<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    match stream.alloc_zeros::<T>(len) {
        Ok(slice) => Ok(slice),
        Err(_) => {
            drain_and_trim()?;
            stream.alloc_zeros::<T>(len)
        }
    }
}

/// Device VRAM budget in bytes for table session admission control.
///
/// LAMBDA_VM_VRAM_BUDGET_MB overrides it (used to force the throttle in tests).
/// Otherwise it is 80% of total device memory, leaving headroom for the
/// context, module code, and retained pool blocks. Returns u64::MAX on any
/// query failure, which disables budgeting (chunks fall back to the core bound
/// size alone).
fn detect_vram_budget_bytes(ctx: &CudaContext) -> u64 {
    if let Ok(mb) = std::env::var("LAMBDA_VM_VRAM_BUDGET_MB")
        && let Ok(mb) = mb.parse::<u64>()
    {
        return mb.saturating_mul(1024 * 1024);
    }
    use cudarc::driver::sys;
    // SAFETY: raw driver query writing into two stack slots. The caller's
    // context is already current (it was just created in `init`). Any error
    // falls through to the budgeting-disabled sentinel.
    unsafe {
        let _ = ctx;
        let mut free: usize = 0;
        let mut total: usize = 0;
        if sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize)
            .result()
            .is_err()
        {
            return u64::MAX;
        }
        // 80% of total, computed to avoid intermediate overflow.
        (total as u64) / 5 * 4
    }
}

impl Backend {
    fn init() -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        // cudarc's default per-slice CudaEvent tracking adds two driver calls
        // per alloc and serialises under the context lock. Cross-stream
        // read-after-write on shared handles is ordered explicitly instead:
        // producers either host-synchronise before the handle escapes (trace
        // snapshot, resident LogUp aux) or attach a `ready` PooledEvent that
        // every consumer awaits via `wait_ready_on` (the R1 LDE handles).
        // Any new cross-stream consumer MUST follow one of those two
        // patterns; with that upheld the tracking is pure overhead.
        unsafe { ctx.disable_event_tracking() };

        // Retain freed device memory in the stream ordered pool for reuse.
        //
        // cudarc routes CudaStream::alloc* through cuMemAllocAsync, drawing from
        // the device default memory pool. Its release threshold defaults to 0,
        // so every freed buffer goes back to the OS at the next sync and the
        // prover's large LDE/FRI buffers are rebuilt from scratch each op.
        // Raising the threshold keeps freed blocks in the pool so a same size
        // allocation skips a real driver allocation. Best effort: on any error
        // we keep the current behaviour.
        retain_default_mempool(&ctx);

        let arith = ctx.load_module(Ptx::from_binary(ARITH_CUBIN.to_vec()))?;
        let ntt = ctx.load_module(Ptx::from_binary(NTT_CUBIN.to_vec()))?;
        let keccak = ctx.load_module(Ptx::from_binary(KECCAK_CUBIN.to_vec()))?;
        let rpx = ctx.load_module(Ptx::from_binary(RPX_CUBIN.to_vec()))?;
        let bary = ctx.load_module(Ptx::from_binary(BARY_CUBIN.to_vec()))?;
        let deep = ctx.load_module(Ptx::from_binary(DEEP_CUBIN.to_vec()))?;
        let fri = ctx.load_module(Ptx::from_binary(FRI_CUBIN.to_vec()))?;
        let inverse = ctx.load_module(Ptx::from_binary(INVERSE_CUBIN.to_vec()))?;
        let logup = ctx.load_module(Ptx::from_binary(LOGUP_CUBIN.to_vec()))?;
        let constraint_interp =
            ctx.load_module(Ptx::from_binary(CONSTRAINT_INTERP_CUBIN.to_vec()))?;
        let blake3 = ctx.load_module(Ptx::from_binary(BLAKE3_CUBIN.to_vec()))?;
        let sumcheck = ctx.load_module(Ptx::from_binary(SUMCHECK_CUBIN.to_vec()))?;
        let whir_fold = ctx.load_module(Ptx::from_binary(WHIR_FOLD_CUBIN.to_vec()))?;

        let mut streams = Vec::with_capacity(STREAM_POOL_SIZE);
        for _ in 0..STREAM_POOL_SIZE {
            streams.push(ctx.new_stream()?);
        }
        // One slot per rayon worker. `current_thread_index()` returns
        // `0..current_num_threads()`, and non-rayon callers (None) map to slot 0,
        // so this many slots covers every caller.
        //
        // `current_num_threads()` returns the default-pool size (the cpu count)
        // when no custom pool is in use. Stable across the backend's lifetime
        // since rayon's pool is fixed at first use.
        let n_slots = rayon::current_num_threads().max(1);
        // Pre-create each slot's reusable event here, off the prove's critical
        // path — a mid-prove cuEventCreate convoys the driver lock (~30 ms
        // measured under load vs ~µs at init).
        let make_pool = || -> Result<Vec<Mutex<PinnedStaging>>> {
            let mut pool = Vec::with_capacity(n_slots);
            for _ in 0..n_slots {
                let mut slot = PinnedStaging::empty();
                slot.event = Some(ctx.new_event(None)?);
                pool.push(Mutex::new(slot));
            }
            Ok(pool)
        };
        let pinned_staging = make_pool()?;
        let pinned_hashes = make_pool()?;
        // Pre-create the handle-readiness event pool (see `take_event`): one
        // event per device-resident handle a prove can have alive; creation
        // here is ~µs each, mid-prove it convoys the driver lock.
        let event_pool = {
            let mut pool = Vec::with_capacity(512);
            for _ in 0..512 {
                pool.push(ctx.new_event(None)?);
            }
            Mutex::new(pool)
        };
        // Separate "utility" stream for twiddle uploads and other bookkeeping;
        // not part of the pool that callers rotate through.
        let util_stream = ctx.new_stream()?;

        // Cache is indexed by log_n. Valid range is [0, TWO_ADICITY] since
        // Goldilocks has roots of unity for orders 2^0..=2^TWO_ADICITY only.
        // Length = TWO_ADICITY + 1 to allow indexing at log_n = TWO_ADICITY.
        let max_log = GoldilocksField::TWO_ADICITY as usize + 1;

        let vram_budget_bytes = detect_vram_budget_bytes(&ctx);

        Ok(Self {
            vector_add_u64: arith.load_function("vector_add_u64")?,
            gl_add: arith.load_function("gl_add_kernel")?,
            gl_sub: arith.load_function("gl_sub_kernel")?,
            gl_mul: arith.load_function("gl_mul_kernel")?,
            gl_neg: arith.load_function("gl_neg_kernel")?,
            ext3_mul: arith.load_function("ext3_mul_kernel")?,
            ext3_add: arith.load_function("ext3_add_kernel")?,
            ext3_sub: arith.load_function("ext3_sub_kernel")?,
            bit_reverse_permute: ntt.load_function("bit_reverse_permute")?,
            lift_spread: ntt.load_function("lift_spread")?,
            mobius_level: ntt.load_function("mobius_level")?,
            mobius_low_levels: ntt.load_function("mobius_low_levels")?,
            mobius_tile: ntt.load_function("mobius_tile")?,
            ntt_dit_level: ntt.load_function("ntt_dit_level")?,
            ntt_dit_8_levels: ntt.load_function("ntt_dit_8_levels")?,
            ntt_dit_tile: ntt.load_function("ntt_dit_tile")?,
            pointwise_mul: ntt.load_function("pointwise_mul")?,
            scalar_mul: ntt.load_function("scalar_mul")?,
            bit_reverse_permute_batched: ntt.load_function("bit_reverse_permute_batched")?,
            ntt_dit_level_batched: ntt.load_function("ntt_dit_level_batched")?,
            ntt_dit_8_levels_batched: ntt.load_function("ntt_dit_8_levels_batched")?,
            pointwise_mul_batched: ntt.load_function("pointwise_mul_batched")?,
            scalar_mul_batched: ntt.load_function("scalar_mul_batched")?,
            bit_reverse_row_major: ntt.load_function("bit_reverse_row_major")?,
            ntt_dit_level_row_major: ntt.load_function("ntt_dit_level_row_major")?,
            ntt_dit_8_levels_row_major: ntt.load_function("ntt_dit_8_levels_row_major")?,
            pointwise_mul_row_major: ntt.load_function("pointwise_mul_row_major")?,
            matrix_transpose_strided: ntt.load_function("matrix_transpose_strided")?,
            keccak256_leaves_base_row_major_row_pair: keccak
                .load_function("keccak256_leaves_base_row_major_row_pair")?,
            keccak256_leaves_base_row_major_row_pair_range: keccak
                .load_function("keccak256_leaves_base_row_major_row_pair_range")?,
            keccak256_leaves_base_batched: keccak.load_function("keccak256_leaves_base_batched")?,
            keccak256_leaves_base_coset: keccak.load_function("keccak256_leaves_base_coset")?,
            keccak256_leaves_ext3_coset: keccak.load_function("keccak256_leaves_ext3_coset")?,
            keccak256_leaves_base_row_pair_batched: keccak
                .load_function("keccak256_leaves_base_row_pair_batched")?,
            keccak256_leaves_ext3_batched: keccak.load_function("keccak256_leaves_ext3_batched")?,
            grind_search: keccak.load_function("grind_search")?,
            keccak_comp_poly_leaves_ext3: keccak.load_function("keccak_comp_poly_leaves_ext3")?,
            keccak_fri_leaves_ext3: keccak.load_function("keccak_fri_leaves_ext3")?,
            keccak_merkle_level: keccak.load_function("keccak_merkle_level")?,
            keccak_merkle_tail: keccak.load_function("keccak_merkle_tail")?,
            merkle_gather_paths: keccak.load_function("merkle_gather_paths")?,
            blake3_leaves_base_row_major_row_pair: blake3
                .load_function("blake3_leaves_base_row_major_row_pair")?,
            blake3_leaves_base_row_major_row_pair_range: blake3
                .load_function("blake3_leaves_base_row_major_row_pair_range")?,
            blake3_leaves_base_batched: blake3.load_function("blake3_leaves_base_batched")?,
            blake3_leaves_base_row_pair_batched: blake3
                .load_function("blake3_leaves_base_row_pair_batched")?,
            blake3_leaves_ext3_batched: blake3.load_function("blake3_leaves_ext3_batched")?,
            blake3_comp_poly_leaves_ext3: blake3.load_function("blake3_comp_poly_leaves_ext3")?,
            blake3_fri_leaves_ext3: blake3.load_function("blake3_fri_leaves_ext3")?,
            blake3_merkle_level: blake3.load_function("blake3_merkle_level")?,
            blake3_merkle_tail: blake3.load_function("blake3_merkle_tail")?,
            blake3_compress_probe_6r: blake3.load_function("blake3_compress_probe_6r")?,
            blake3_compress_probe_7r: blake3.load_function("blake3_compress_probe_7r")?,
            blake3_compress_probe_default: blake3.load_function("blake3_compress_probe_default")?,
            blake3_rounds_probe: blake3.load_function("blake3_rounds_probe")?,
            blake3_serialize_felts_probe: blake3.load_function("blake3_serialize_felts_probe")?,
            blake3_blocks_of_felts_probe: blake3.load_function("blake3_blocks_of_felts_probe")?,
            blake3_chain_probe: blake3.load_function("blake3_chain_probe")?,

            rpx_leaves_base_row_major_row_pair: rpx
                .load_function("rpx_leaves_base_row_major_row_pair")?,
            rpx_leaves_base_row_major_row_pair_range: rpx
                .load_function("rpx_leaves_base_row_major_row_pair_range")?,
            rpx_leaves_base_batched: rpx.load_function("rpx_leaves_base_batched")?,
            rpx_leaves_base_row_pair_batched: rpx
                .load_function("rpx_leaves_base_row_pair_batched")?,
            rpx_leaves_ext3_batched: rpx.load_function("rpx_leaves_ext3_batched")?,
            rpx_comp_poly_leaves_ext3: rpx.load_function("rpx_comp_poly_leaves_ext3")?,
            rpx_fri_leaves_ext3: rpx.load_function("rpx_fri_leaves_ext3")?,
            rpx_merkle_level: rpx.load_function("rpx_merkle_level")?,
            rpx_merkle_tail: rpx.load_function("rpx_merkle_tail")?,
            rpx_permute_probe: rpx.load_function("rpx_permute_probe")?,
            // The WHIR path's leaves are a fold COSET, not a row group, so these
            // two are additional kernels rather than alternatives to the seven
            // above. `rpx_merkle_level` / `rpx_merkle_tail` are shared.
            rpx_leaves_base_coset: rpx.load_function("rpx_leaves_base_coset")?,
            rpx_leaves_ext3_coset: rpx.load_function("rpx_leaves_ext3_coset")?,
            rpx_grind_search: rpx.load_function("rpx_grind_search")?,
            rpx_grind_search_counted: rpx.load_function("rpx_grind_search_counted")?,
            barycentric_base_batched: bary.load_function("barycentric_base_batched")?,
            barycentric_ext3_batched: bary.load_function("barycentric_ext3_batched")?,
            barycentric_base_batched_strided: bary
                .load_function("barycentric_base_batched_strided")?,
            barycentric_ext3_batched_strided: bary
                .load_function("barycentric_ext3_batched_strided")?,
            barycentric_base_strided_multi: bary.load_function("barycentric_base_strided_multi")?,
            barycentric_ext3_strided_multi: bary.load_function("barycentric_ext3_strided_multi")?,
            barycentric_combine_partials: bary.load_function("barycentric_combine_partials")?,
            gather_rows_base: bary.load_function("gather_rows_base")?,
            gather_rows_ext3: bary.load_function("gather_rows_ext3")?,
            deep_composition_ext3_row: deep.load_function("deep_composition_ext3_row")?,
            bit_reverse_ext3_kernel: deep.load_function("bit_reverse_ext3_interleaved")?,
            fri_fold_ext3: fri.load_function("fri_fold_ext3")?,
            gather_ext3_at: fri.load_function("gather_ext3_at")?,
            fri_update_twiddles: fri.load_function("fri_update_twiddles")?,
            compute_denoms_ext3: inverse.load_function("compute_denoms_ext3")?,
            block_inclusive_scan_fwd_ext3: inverse
                .load_function("block_inclusive_scan_fwd_ext3")?,
            apply_block_offsets_fwd_ext3: inverse.load_function("apply_block_offsets_fwd_ext3")?,
            block_inclusive_scan_rev_ext3: inverse
                .load_function("block_inclusive_scan_rev_ext3")?,
            apply_block_offsets_rev_ext3: inverse.load_function("apply_block_offsets_rev_ext3")?,
            batch_inverse_combine_ext3: inverse.load_function("batch_inverse_combine_ext3")?,
            invert_total_ext3: inverse.load_function("invert_total_ext3")?,
            logup_fingerprint_ext3: logup.load_function("logup_fingerprint_ext3")?,
            logup_term_ext3: logup.load_function("logup_term_ext3")?,
            logup_row_sum_ext3: logup.load_function("logup_row_sum_ext3")?,
            logup_scan_block_add_ext3: logup.load_function("logup_scan_block_add_ext3")?,
            logup_apply_offsets_add_ext3: logup.load_function("logup_apply_offsets_add_ext3")?,
            logup_finalize_accum_ext3: logup.load_function("logup_finalize_accum_ext3")?,
            logup_assemble_aux_ext3: logup.load_function("logup_assemble_aux_ext3")?,
            whir_fold_base_ext3: whir_fold.load_function("whir_fold_base_ext3")?,
            whir_fold_ext3: whir_fold.load_function("whir_fold_ext3")?,
            gather_cosets: whir_fold.load_function("gather_cosets")?,
            sumcheck_round_ext3: sumcheck.load_function("sumcheck_round_ext3")?,
            sum_partials_ext3: sumcheck.load_function("sum_partials_ext3")?,
            sumcheck_fold_ext3: sumcheck.load_function("sumcheck_fold_ext3")?,
            mle_fold_base_ext3: sumcheck.load_function("mle_fold_base_ext3")?,
            eq_expand_level_ext3: sumcheck.load_function("eq_expand_level_ext3")?,
            eq_seed_shares_ext3: sumcheck.load_function("eq_seed_shares_ext3")?,
            eq_expand_level_shares_ext3: sumcheck.load_function("eq_expand_level_shares_ext3")?,
            program_map_ext3: sumcheck.load_function("program_map_ext3")?,
            factors_from_columns_ext3: sumcheck.load_function("factors_from_columns_ext3")?,
            mle_lift_base_ext3: sumcheck.load_function("mle_lift_base_ext3")?,
            add_scaled_ext3: sumcheck.load_function("add_scaled_ext3")?,
            fill_ext3: sumcheck.load_function("fill_ext3")?,
            fraction_fold_ext3: sumcheck.load_function("fraction_fold_ext3")?,
            fraction_fold_padded_ext3: sumcheck.load_function("fraction_fold_padded_ext3")?,
            mle_fold_base_ext3_many: sumcheck.load_function("mle_fold_base_ext3_many")?,
            constraint_interp_kernel: constraint_interp
                .load_function("constraint_interp_kernel")?,
            constraint_composition_kernel: constraint_interp
                .load_function("constraint_composition_kernel")?,
            decompose_d2_kernel: constraint_interp.load_function("decompose_d2_ext3")?,
            comp_h_to_slabs_kernel: constraint_interp.load_function("comp_h_to_slabs_ext3")?,
            fwd_twiddles: Mutex::new(vec![None; max_log]),
            inv_twiddles: Mutex::new(vec![None; max_log]),
            ctx,
            streams,
            pinned_staging,
            pinned_hashes,
            event_pool,
            util_stream,
            next: AtomicUsize::new(0),
            vram_budget_bytes,
            reserved: AtomicU64::new(0),
        })
    }

    /// VRAM budget in bytes for admission control. `u64::MAX`
    /// when budgeting is disabled (query failed). See the field docs.
    pub fn vram_budget_bytes(&self) -> u64 {
        self.vram_budget_bytes
    }

    /// Live `(free, total)` device memory in bytes, for diagnostics — the
    /// admission gates never read it (they must answer the same at R1 and at
    /// R4). `None` when the query fails.
    pub fn device_mem_info(&self) -> Option<(u64, u64)> {
        self.ctx
            .mem_get_info()
            .ok()
            .map(|(free, total)| (free as u64, total as u64))
    }

    /// Hand the default memory pool's unused reserved memory back to the OS,
    /// keeping at most `keep_bytes` (`cuMemPoolTrimTo`). Under the retained
    /// posture ([`mempool_release_threshold_bytes`]) a sync never releases;
    /// this is the explicit release for the moments reuse cannot serve — a
    /// differently-shaped table after a large one, or a sampler that must read
    /// the live working set. Best effort: `false` when the pool cannot be
    /// queried or the trim fails.
    pub fn trim_mempool_to(&self, keep_bytes: u64) -> bool {
        use cudarc::driver::sys;
        // SAFETY: raw driver calls on this backend's live context; the trim
        // takes a plain byte count.
        unsafe {
            default_mempool(&self.ctx).is_some_and(|pool| {
                sys::cuMemPoolTrimTo(pool, keep_bytes as usize)
                    .result()
                    .is_ok()
            })
        }
    }

    /// Bytes the device reports free, right now.
    ///
    /// The driver's own accounting rather than this module's: [`reserve`]
    /// counts what callers PROMISED, which is silent about anything allocated
    /// without a reservation. A test that wants to know whether a structure is
    /// holding device memory it never declared has to ask the device, and this
    /// is how — see `a_group_holds_only_its_codewords_before_any_open`.
    ///
    /// ⚠ The stream-ordered pool retains freed blocks, so this falls as memory
    /// is used and does not always rise as it is released. It answers "how
    /// much has this process taken from the device", not "how much is live",
    /// which is the question a retention test is asking.
    pub fn free_vram_bytes(&self) -> Result<u64> {
        use cudarc::driver::sys;
        self.ctx.bind_to_thread()?;
        // SAFETY: a raw driver query writing into two stack slots, with the
        // context bound to this thread on the line above.
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize).result()?;
            Ok(free as u64)
        }
    }

    /// Promises `bytes` of the device to something about to be built there, or
    /// refuses.
    ///
    /// Asking the driver how much is free does not answer this: two callers
    /// can both be told yes and both be right at the moment they ask. What a
    /// structure needs is that the room stays its own until it is done — a
    /// codeword that is admitted and then cannot fold has nowhere to go, since
    /// there is no copy on the host by then.
    ///
    /// Refusing is cheap wherever it is asked, because everything that asks
    /// has a host path. The budget binds only when several proofs share a
    /// card: one of them is enough to fill it.
    pub fn reserve(&self, bytes: u64) -> Option<DeviceReservation> {
        let mut held = self.reserved.load(Ordering::Relaxed);
        loop {
            if held.saturating_add(bytes) > self.vram_budget_bytes {
                return None;
            }
            match self.reserved.compare_exchange_weak(
                held,
                held + bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(DeviceReservation {
                        bytes: AtomicU64::new(bytes),
                    });
                }
                Err(seen) => held = seen,
            }
        }
    }

    /// Bytes promised across every live reservation — what `reserve` checks
    /// the budget against. Exposed so a test can assert the number rather than
    /// assert that nothing crashed.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved.load(Ordering::Relaxed)
    }

    /// Round-robin over the stream pool. Concurrent callers get different
    /// streams so their kernel launches overlap on the GPU.
    pub fn next_stream(&self) -> Arc<CudaStream> {
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        self.streams[idx].clone()
    }

    /// Per-rayon-worker pinned staging buffer. Returns the slot for the
    /// current worker (or slot 0 outside a rayon context). Grows lazily to
    /// the largest LDE the worker has seen. See [`Backend`]'s
    /// `pinned_staging` field for the rationale behind the per-worker
    /// split.
    pub fn pinned_staging(&self) -> &Mutex<PinnedStaging> {
        &self.pinned_staging[self.worker_slot(self.pinned_staging.len())]
    }

    /// Per-worker pinned staging for Merkle leaf hash output. Sized in u64
    /// units. Caller should reserve `(num_rows * 32 + 7) / 8` u64s.
    pub fn pinned_hashes(&self) -> &Mutex<PinnedStaging> {
        &self.pinned_hashes[self.worker_slot(self.pinned_hashes.len())]
    }

    /// Map `rayon::current_thread_index()` to a slot index, with a defensive
    /// clamp in case the rayon pool grew past the Vec we sized at init.
    ///
    /// The per-table scheduler's driver threads are not rayon workers: they
    /// all resolve to slot 0 and deliberately share one slab. Spreading them
    /// over per-driver slots costs more in repeated pinned allocation than
    /// the shared mutex does — the staged transfers are already hidden by
    /// cross-table overlap.
    fn worker_slot(&self, len: usize) -> usize {
        let idx = rayon::current_thread_index().unwrap_or(0);
        // Should be unreachable with rayon's fixed default pool, but if a
        // larger custom pool sneaks in we still want safety: Fall back to
        // slot 0 (correctness preserved, just contention).
        debug_assert!(idx < len, "rayon worker {idx} >= staging slots {len}");
        idx.min(len.saturating_sub(1))
    }

    pub fn fwd_twiddles_for(&self, log_n: u64) -> Result<Arc<CudaSlice<u64>>> {
        self.cached_twiddles(log_n, true)
    }

    pub fn inv_twiddles_for(&self, log_n: u64) -> Result<Arc<CudaSlice<u64>>> {
        self.cached_twiddles(log_n, false)
    }

    fn cached_twiddles(&self, log_n: u64, forward: bool) -> Result<Arc<CudaSlice<u64>>> {
        let idx = log_n as usize;
        let cache = if forward {
            &self.fwd_twiddles
        } else {
            &self.inv_twiddles
        };
        // Cache is sized TWO_ADICITY + 1 in `Backend::init`. Callers derive
        // log_n from `trailing_zeros` of valid Goldilocks domain sizes so it
        // must stay in range; assert in debug to catch regressions.
        debug_assert!(
            log_n <= GoldilocksField::TWO_ADICITY,
            "log_n {log_n} exceeds Goldilocks TWO_ADICITY ({})",
            GoldilocksField::TWO_ADICITY,
        );
        {
            let guard = cache.lock().unwrap();
            if let Some(t) = &guard[idx] {
                return Ok(t.clone());
            }
        }
        // Compute on host, upload on the utility stream. Another thread may
        // have populated the cache in the meantime; prefer that entry.
        let host = if forward {
            twiddles_forward(log_n)
        } else {
            twiddles_inverse(log_n)
        };
        let dev = Arc::new(self.util_stream.clone_htod(&host)?);
        self.util_stream.synchronize()?;
        let mut guard = cache.lock().unwrap();
        if let Some(t) = &guard[idx] {
            Ok(t.clone())
        } else {
            guard[idx] = Some(dev.clone());
            Ok(dev)
        }
    }
}

/// Returns the process-wide CUDA backend, initialising it on first call.
///
/// Returns `Err` when CUDA initialisation fails (no driver, no GPU, PTX load
/// failure). Initialisation is retried on the next call until one succeeds —
/// only a successful `Backend` is cached. The race window where two threads
/// init concurrently is harmless: at most one extra `Backend::init()` runs
/// and the loser is dropped.
pub fn backend() -> Result<&'static Backend> {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = match Backend::init() {
        Ok(b) => b,
        Err(e) => {
            // Backend init failing means every GPU entry point silently falls
            // back to CPU. That is expected on a GPU-less host, but it also
            // fires when the AOT cubins won't load — most often a build-host vs
            // run-host GPU-arch mismatch (cubins are compiled for the detected
            // `sm_XX`) or an empty nvcc-less stub. Warn once so the fallback is
            // never silent: rebuild on the run host, or set `CUDARC_NVCC_ARCH`.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                eprintln!(
                    "math-cuda: GPU backend unavailable ({e}) — running on CPU. \
                     If a GPU is present this is likely a kernel-cubin arch mismatch; \
                     rebuild on the run host or set CUDARC_NVCC_ARCH to its sm_XX."
                );
            });
            return Err(e);
        }
    };
    let _ = BACKEND.set(b);
    Ok(BACKEND.get().expect("backend just initialised"))
}

// ── Asynchronous D2H through the pinned staging slabs ────────────────────────

/// A device→host copy enqueued into a per-worker pinned staging slab, not yet
/// awaited. Created by [`async_dtoh_via`]; consumed by one of the `wait_*`
/// methods, which block only until the copy (and everything queued before it
/// on its stream) lands.
///
/// Holding this value keeps the staging slot's mutex locked, which is what
/// makes the whole scheme safe: no other caller (and no capacity growth) can
/// touch the slab while the DMA is in flight. Corollary: never call
/// `htod_via`/`async_dtoh_via` on the same slot from the thread holding a
/// live `PendingD2H` — the non-reentrant slot mutex self-deadlocks.
pub struct PendingD2H<'a> {
    staging: std::sync::MutexGuard<'a, PinnedStaging>,
    n_bytes: usize,
}

// A dropped pending (e.g. a `?` between enqueue and wait) must not release
// the slot while the DMA is still writing the slab: the next holder could
// repack it or `ensure_capacity` could free it mid-copy. Block on the copy's
// event before the guard drops; errors are ignored (the context is already
// failing on these paths, and the wait is best-effort protection).
impl Drop for PendingD2H<'_> {
    fn drop(&mut self) {
        let _ = self.staging.sync_event();
    }
}

/// Chunk size for [`htod_via`]'s staged upload — the upper bound a single H2D
/// puts on a staging slot's page-locked footprint. 64 MB is large enough to
/// amortize the per-chunk DMA launch + event sync, small enough to keep the
/// pinned slab independent of trace size.
const HTOD_CHUNK_BYTES: usize = 64 << 20; // 64 MB

/// Host→device copy staged through the pinned slot, in fixed-size chunks: each
/// chunk is one host memcpy into pinned memory + one async DMA, instead of the
/// driver's internal pageable staging (small chunks; 2-3x slower for
/// multi-hundred-MB traces and it convoys under multi-thread load). Blocks
/// until the last DMA lands, so the slot and `src_host` are both reusable on
/// return.
///
/// Chunking caps the slot's page-locked footprint at [`HTOD_CHUNK_BYTES`]
/// regardless of trace size. This matters on the device-only path
/// (`retain_host_lde = false`): there is no [`async_dtoh_via`] drain to size
/// the slot, so `htod_via` is its only writer — an uncapped copy would grow
/// the per-worker slab to a whole trace and, being grow-only, never shrink it.
/// The host-retaining path is unaffected: its later `async_dtoh_via` grows the
/// same slot to the full LDE anyway, and we simply reuse the first chunk of it.
pub fn htod_via<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    slot: &Mutex<PinnedStaging>,
    ctx: &CudaContext,
    src_host: &[T],
    dst: &mut cudarc::driver::CudaViewMut<'_, T>,
) -> Result<()> {
    use cudarc::driver::DevicePtrMut;
    assert!(
        dst.len() >= src_host.len(),
        "htod_via: destination shorter than source"
    );
    let n_bytes = std::mem::size_of_val(src_host);
    if n_bytes == 0 {
        return Ok(());
    }
    let elem_size = std::mem::size_of::<T>();
    // Chunk in whole elements so a `T` never straddles a chunk boundary.
    let chunk_elems = (HTOD_CHUNK_BYTES / elem_size.max(1)).max(1);

    let mut staging = slot.lock().unwrap();
    // Only ask for a chunk's worth of pinned memory (or the whole copy when
    // smaller). If another path (`async_dtoh_via` on the host-retaining flow)
    // already grew this slot larger, it stays larger — grow-only — and we just
    // use the first chunk of it.
    let want_u64 = (chunk_elems * elem_size)
        .div_ceil(8)
        .min(n_bytes.div_ceil(8));
    staging.ensure_capacity(want_u64, ctx)?;
    ctx.bind_to_thread()?;

    // SAFETY: `device_ptr_mut` yields the destination base pointer and orders
    // the device writes on `stream`; `dst.len() >= src_host.len()` (asserted),
    // so every chunk's byte range stays within `dst`.
    let (dst_base, _record) = dst.device_ptr_mut(stream);
    // Declared after the slot's MutexGuard so it drops FIRST: once a chunk's
    // DMA is in flight, any `?`-return below must drain the stream before the
    // guard releases the slot, or the next locker's `ensure_capacity` could
    // `cuMemFreeHost` the slab while the device is still reading it. Same
    // hazard `async_dtoh_via` guards against on its record-event failure.
    let mut drain = DrainOnErr {
        stream,
        armed: false,
    };
    let src = src_host.as_ptr() as *const u8;
    let n_elems = src_host.len();
    let mut elem_off = 0usize;
    while elem_off < n_elems {
        let this_elems = (n_elems - elem_off).min(chunk_elems);
        let this_bytes = this_elems * elem_size;
        let byte_off = elem_off * elem_size;
        // SAFETY: the pinned slab holds at least `chunk_elems * elem_size`
        // bytes (or the whole copy when smaller). The previous chunk's DMA is
        // synced below before this memcpy overwrites the slab, so the slab is
        // never read (by an in-flight DMA) and written at the same time.
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(byte_off), staging.ptr as *mut u8, this_bytes);
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst_base + byte_off as u64,
                staging.ptr as *const core::ffi::c_void,
                this_bytes,
                stream.cu_stream(),
            )
            .result();
            // Armed even on failure: the driver may have enqueued the copy
            // before reporting the error.
            drain.armed = true;
            r?;
        }
        // Single-buffered: wait for this chunk's DMA before the next memcpy
        // reuses the slab. Both calls can fail with the DMA still in flight,
        // which is what `drain` covers.
        staging.record_event(stream)?;
        staging.sync_event()?;
        // This chunk has landed; nothing is reading the slab until the next
        // iteration re-arms.
        drain.armed = false;
        elem_off += this_elems;
    }
    Ok(())
}

/// Enqueue an async D2H of `n_elems` of `src` into the pinned slab of `slot`,
/// without synchronizing the stream. Unlike `stream.memcpy_dtoh` into a plain
/// (pageable) slice — which the driver services synchronously — this returns
/// as soon as the copy is queued; the returned [`PendingD2H`] is awaited at
/// the point the host actually needs the bytes.
///
/// SAFETY contract (upheld by construction for our callers): `src` must stay
/// alive until the copy completes. Dropping a `CudaSlice` frees it
/// stream-ordered on its own stream, so a `src` allocated on `stream` may be
/// dropped after this call — the free queues behind the copy. Do NOT pass a
/// `src` owned by a *different* stream and drop it before waiting.
pub fn async_dtoh_via<'a, T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    slot: &'a Mutex<PinnedStaging>,
    ctx: &CudaContext,
    src: &CudaSlice<T>,
    n_elems: usize,
) -> Result<PendingD2H<'a>> {
    use cudarc::driver::DevicePtr;
    assert!(n_elems <= src.len());
    let n_bytes = n_elems * std::mem::size_of::<T>();
    let u64_len = n_bytes.div_ceil(8);
    let mut staging = slot.lock().unwrap();
    staging.ensure_capacity(u64_len, ctx)?;
    ctx.bind_to_thread()?;
    // SAFETY: dst is this slot's pinned allocation — stable address (only
    // `ensure_capacity` moves it, and we hold the lock), pinned (registered
    // via cuMemHostAlloc, so the driver DMAs directly, asynchronously).
    // `device_ptr` orders the read after prior writes on `stream`.
    unsafe {
        let (src_ptr, _record) = src.device_ptr(stream);
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
            staging.ptr as *mut core::ffi::c_void,
            src_ptr,
            n_bytes,
            stream.cu_stream(),
        )
        .result()?;
    }
    // Re-record the slot's reusable event (created once — per-call
    // cuEventCreate/Destroy convoys the driver lock under load). The DMA is
    // already in flight: if the record fails, drain the stream before the
    // guard drops, or the next locker could free the slab mid-copy.
    if let Err(e) = staging.record_event(stream) {
        let _ = stream.synchronize();
        return Err(e);
    }
    Ok(PendingD2H { staging, n_bytes })
}

/// Best-effort stream drain on error paths: while `armed`, dropping this guard
/// synchronizes the stream. Arm after the first enqueue that reads a
/// pinned-staging slab; defuse once the slab is safe to release. Declared
/// AFTER the slot's `MutexGuard`, it drops first, so an `?`-return can never
/// release the slot with a DMA still reading it.
pub(crate) struct DrainOnErr<'a> {
    pub stream: &'a CudaStream,
    pub armed: bool,
}

impl Drop for DrainOnErr<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.stream.synchronize();
        }
    }
}

impl PendingD2H<'_> {
    /// Number of bytes the copy deposits.
    pub fn len_bytes(&self) -> usize {
        self.n_bytes
    }

    /// Block until the copy lands, then read the pinned bytes through `f`.
    /// Consumes the pending (releasing the staging slot when `f` returns).
    pub fn wait_and_read<R>(self, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        self.staging
            .event
            .as_ref()
            .expect("recorded by async_dtoh_via")
            .synchronize()?;
        // SAFETY: event completion orders the DMA before this read; the slab
        // is exclusively ours while the guard lives.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.staging.ptr as *const u8, self.n_bytes) };
        Ok(f(bytes))
    }

    /// Wait and copy the bytes out into `dst` (pageable is fine — this is a
    /// plain host memcpy at RAM speed, not a DMA target).
    pub fn wait_into_bytes(self, dst: &mut [u8]) -> Result<()> {
        assert_eq!(dst.len(), self.n_bytes);
        self.wait_and_read(|src| dst.copy_from_slice(src))
    }

    /// Wait and copy out as u64s. `dst.len() * 8` must equal the copied bytes.
    pub fn wait_into_u64(self, dst: &mut [u64]) -> Result<()> {
        assert_eq!(dst.len() * 8, self.n_bytes);
        self.staging
            .event
            .as_ref()
            .expect("recorded by async_dtoh_via")
            .synchronize()?;
        // SAFETY: as in `wait_and_read`; the slab is u64-aligned by
        // construction.
        let src = unsafe { std::slice::from_raw_parts(self.staging.ptr as *const u64, dst.len()) };
        dst.copy_from_slice(src);
        Ok(())
    }
}

// ── Pooled events for handle-readiness tracking ──────────────────────────────

/// A pre-created CUDA event borrowed from the backend's free-list; returns
/// itself to the list on drop. Used as the `ready` marker on device-resident
/// handles (`GpuLdeBase`/`GpuLdeExt3`) so consumers on other streams can wait
/// device-side (`stream.wait`) instead of the producer host-blocking in a
/// final synchronize. Pooled because a mid-prove cuEventCreate convoys the
/// driver lock (see `PinnedStaging::record_event`).
pub struct PooledEvent {
    event: Option<cudarc::driver::CudaEvent>,
}

impl PooledEvent {
    pub fn event(&self) -> &cudarc::driver::CudaEvent {
        self.event.as_ref().expect("present until drop")
    }
}

impl Drop for PooledEvent {
    fn drop(&mut self) {
        if let (Some(ev), Ok(be)) = (self.event.take(), backend()) {
            be.event_pool.lock().unwrap().push(ev);
        }
    }
}

impl Backend {
    /// Take a pre-created event from the pool (creating one only if the pool
    /// ran dry, which should not happen in a normal prove).
    pub fn take_event(&self) -> Result<PooledEvent> {
        let ev = match self.event_pool.lock().unwrap().pop() {
            Some(ev) => ev,
            None => self.ctx.new_event(None)?,
        };
        Ok(PooledEvent { event: Some(ev) })
    }
}

#[cfg(test)]
mod device_fallback_counter_tests {
    use super::{device_fallbacks, note_device_fallback, reset_device_fallbacks};
    use std::sync::Mutex;

    /// The counter is process-wide, so a test asserting an absolute value
    /// serialises against anything else in this binary that might move it.
    static COUNTER: Mutex<()> = Mutex::new(());

    /// The plumbing, card-free: a bump reads as one, bumps accumulate, and a
    /// reset reads as zero. This is the CONTROL for the box test — if
    /// `note`/`device_fallbacks`/`reset` did not agree here, no site test could
    /// be trusted.
    #[test]
    fn note_bumps_read_and_reset_zeroes() {
        let _g = COUNTER.lock().unwrap_or_else(|e| e.into_inner());
        reset_device_fallbacks();
        assert_eq!(device_fallbacks(), 0, "reset must zero the counter");
        note_device_fallback();
        assert_eq!(device_fallbacks(), 1, "one note reads one");
        note_device_fallback();
        assert_eq!(device_fallbacks(), 2, "notes accumulate");
        reset_device_fallbacks();
        assert_eq!(device_fallbacks(), 0, "reset must zero it again");
    }

    /// ⛔ THE FOUR UNDRIVEN SITES' FALSIFIER. The box test drives ONE site
    /// (`DeviceColumns::upload`); this arm is what lets the other four fail
    /// without four card fixtures. `note_device_fallback` must be CALLED at
    /// exactly the five argue-surface sites the counter's doc enumerates —
    /// three in `sumcheck.rs`, one in `gkr.rs`, one in `columns.rs`. Removing
    /// the call at ANY site changes the tuple and reddens this test by name,
    /// which localises the loss to the file it happened in.
    ///
    /// The pattern carries the `crate::device::` prefix so it counts CALLS and
    /// never the definition (`pub(crate) fn note_device_fallback`).
    #[test]
    fn note_device_fallback_is_called_at_exactly_the_five_argue_sites() {
        const PATTERN: &str = "crate::device::note_device_fallback()";
        let sumcheck = include_str!("sumcheck.rs").matches(PATTERN).count();
        let gkr = include_str!("gkr.rs").matches(PATTERN).count();
        let columns = include_str!("columns.rs").matches(PATTERN).count();
        assert_eq!(
            (sumcheck, gkr, columns),
            (3, 1, 1),
            "the argue-surface device-fallback counter must be bumped at exactly \
             the five sites the counter's doc names: sumcheck ×3, gkr ×1, \
             columns ×1 (found sumcheck {sumcheck}, gkr {gkr}, columns {columns})"
        );
    }
}
