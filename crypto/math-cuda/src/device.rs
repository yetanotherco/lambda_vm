//! CUDA device context, stream pool, kernel handles, and twiddle cache.
//!
//! One process-wide backend — lazy-initialised on first use. All kernels live
//! on a single CUDA context; a pool of streams lets rayon-parallel callers
//! overlap H2D / compute / D2H.

use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Pool-wide first-allocation size hint (u64 elems), shared by every slot
    /// of the pool this staging belongs to. Published by the prover once table
    /// sizes are known (see [`Backend::set_staging_size_hints`]); zero = none.
    hint_u64s: Arc<AtomicUsize>,
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
    fn empty(hint_u64s: Arc<AtomicUsize>) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            capacity_elems: 0,
            hint_u64s,
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
        // First allocation of this slot jumps straight to the prove-wide size
        // hint (when the prover published one), so a slot pays cuMemHostAlloc
        // once instead of climbing a realloc ladder (pinned alloc/free is
        // ~50ms+ per step — see docs/gpu_baseline_ethrex_5090.md).
        let target = min_elems.max(self.hint_u64s.load(Ordering::Relaxed));
        let new_cap = target.next_power_of_two().max(1 << 20); // at least 8 MB
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
const BARY_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/barycentric.cubin"));
const DEEP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deep.cubin"));
const FRI_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fri.cubin"));
const INVERSE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inverse.cubin"));
const LOGUP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/logup.cubin"));
const CONSTRAINT_INTERP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/constraint_interp.cubin"));

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
    /// First-allocation size hints for the two staging pools (u64 elems),
    /// shared with every slot. See [`Backend::set_staging_size_hints`].
    staging_hint_u64s: Arc<AtomicUsize>,
    hashes_hint_u64s: Arc<AtomicUsize>,
    util_stream: Arc<CudaStream>,
    /// Free-list of pre-created events for [`Backend::take_event`].
    event_pool: Mutex<Vec<cudarc::driver::CudaEvent>>,
    next: AtomicUsize,
    /// VRAM budget (bytes) for table-session admission control. See
    /// [`detect_vram_budget_bytes`].
    vram_budget_bytes: u64,

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
    pub ntt_dit_level: CudaFunction,
    pub ntt_dit_8_levels: CudaFunction,
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
    pub pointwise_mul_row_major: CudaFunction,
    pub matrix_transpose_strided: CudaFunction,

    // keccak.cubin
    pub keccak256_leaves_base_row_major_row_pair: CudaFunction,
    pub keccak256_leaves_base_row_major_row_pair_range: CudaFunction,
    pub keccak256_leaves_base_batched: CudaFunction,
    pub keccak256_leaves_base_row_pair_batched: CudaFunction,
    pub keccak256_leaves_ext3_batched: CudaFunction,
    pub keccak_comp_poly_leaves_ext3: CudaFunction,
    pub keccak_fri_leaves_ext3: CudaFunction,
    pub keccak_merkle_level: CudaFunction,
    pub merkle_gather_paths: CudaFunction,

    // barycentric.cubin
    pub barycentric_base_batched: CudaFunction,
    pub barycentric_ext3_batched: CudaFunction,
    pub barycentric_base_batched_strided: CudaFunction,
    pub barycentric_ext3_batched_strided: CudaFunction,
    pub gather_rows_base: CudaFunction,
    pub gather_rows_ext3: CudaFunction,

    // deep.cubin
    pub deep_composition_ext3_row: CudaFunction,
    pub bit_reverse_ext3_kernel: CudaFunction,

    // fri.cubin
    pub fri_fold_ext3: CudaFunction,
    pub fri_update_twiddles: CudaFunction,

    // inverse.cubin
    pub compute_denoms_ext3: CudaFunction,
    pub block_inclusive_scan_fwd_ext3: CudaFunction,
    pub apply_block_offsets_fwd_ext3: CudaFunction,
    pub block_inclusive_scan_rev_ext3: CudaFunction,
    pub apply_block_offsets_rev_ext3: CudaFunction,
    pub batch_inverse_combine_ext3: CudaFunction,
    pub logup_fingerprint_ext3: CudaFunction,
    pub logup_term_ext3: CudaFunction,
    pub logup_row_sum_ext3: CudaFunction,
    pub logup_scan_block_add_ext3: CudaFunction,
    pub logup_apply_offsets_add_ext3: CudaFunction,
    pub logup_finalize_accum_ext3: CudaFunction,
    pub logup_assemble_aux_ext3: CudaFunction,

    // constraint_interp.cubin
    pub constraint_interp_kernel: CudaFunction,
    pub constraint_composition_kernel: CudaFunction,
    pub decompose_d2_kernel: CudaFunction,

    // Twiddle caches keyed by log_n.
    fwd_twiddles: Mutex<Vec<Option<Arc<CudaSlice<u64>>>>>,
    inv_twiddles: Mutex<Vec<Option<Arc<CudaSlice<u64>>>>>,
}

/// Raise the device default memory pool's release threshold so freed
/// stream-ordered allocations are kept for reuse instead of returned to the OS
/// at each sync. Best-effort: any failure (e.g. a device/driver without
/// stream-ordered allocator support) leaves the default behaviour untouched.
fn retain_default_mempool(ctx: &CudaContext) {
    use cudarc::driver::sys;
    // SAFETY: raw CUDA driver calls. `ctx.cu_device()` is a valid device for
    // the just-created context; the out-pointers are valid stack slots; the
    // threshold is read as a u64 by the driver. Errors are swallowed.
    unsafe {
        let dev = ctx.cu_device();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool as *mut _, dev)
            .result()
            .is_err()
        {
            return;
        }
        // Default: retain freed stream-ordered blocks indefinitely (u64::MAX)
        // for reuse. `LAMBDA_VM_MEMPOOL_RELEASE_MB` overrides the cap (bytes the
        // pool keeps before returning memory to the OS) when retained-pool
        // growth needs bounding.
        let threshold: u64 = std::env::var("LAMBDA_VM_MEMPOOL_RELEASE_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(u64::MAX);
        let _ = sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *mut core::ffi::c_void,
        )
        .result();
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
        let bary = ctx.load_module(Ptx::from_binary(BARY_CUBIN.to_vec()))?;
        let deep = ctx.load_module(Ptx::from_binary(DEEP_CUBIN.to_vec()))?;
        let fri = ctx.load_module(Ptx::from_binary(FRI_CUBIN.to_vec()))?;
        let inverse = ctx.load_module(Ptx::from_binary(INVERSE_CUBIN.to_vec()))?;
        let logup = ctx.load_module(Ptx::from_binary(LOGUP_CUBIN.to_vec()))?;
        let constraint_interp =
            ctx.load_module(Ptx::from_binary(CONSTRAINT_INTERP_CUBIN.to_vec()))?;

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
        let staging_hint = Arc::new(AtomicUsize::new(0));
        let hashes_hint = Arc::new(AtomicUsize::new(0));
        // Pre-create each slot's reusable event here, off the prove's critical
        // path — a mid-prove cuEventCreate convoys the driver lock (~30 ms
        // measured under load vs ~µs at init).
        let make_pool = |hint: &Arc<AtomicUsize>| -> Result<Vec<Mutex<PinnedStaging>>> {
            let mut pool = Vec::with_capacity(n_slots);
            for _ in 0..n_slots {
                let mut slot = PinnedStaging::empty(Arc::clone(hint));
                slot.event = Some(ctx.new_event(None)?);
                pool.push(Mutex::new(slot));
            }
            Ok(pool)
        };
        let pinned_staging = make_pool(&staging_hint)?;
        let pinned_hashes = make_pool(&hashes_hint)?;
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
            ntt_dit_level: ntt.load_function("ntt_dit_level")?,
            ntt_dit_8_levels: ntt.load_function("ntt_dit_8_levels")?,
            pointwise_mul: ntt.load_function("pointwise_mul")?,
            scalar_mul: ntt.load_function("scalar_mul")?,
            bit_reverse_permute_batched: ntt.load_function("bit_reverse_permute_batched")?,
            ntt_dit_level_batched: ntt.load_function("ntt_dit_level_batched")?,
            ntt_dit_8_levels_batched: ntt.load_function("ntt_dit_8_levels_batched")?,
            pointwise_mul_batched: ntt.load_function("pointwise_mul_batched")?,
            scalar_mul_batched: ntt.load_function("scalar_mul_batched")?,
            bit_reverse_row_major: ntt.load_function("bit_reverse_row_major")?,
            ntt_dit_level_row_major: ntt.load_function("ntt_dit_level_row_major")?,
            pointwise_mul_row_major: ntt.load_function("pointwise_mul_row_major")?,
            matrix_transpose_strided: ntt.load_function("matrix_transpose_strided")?,
            keccak256_leaves_base_row_major_row_pair: keccak
                .load_function("keccak256_leaves_base_row_major_row_pair")?,
            keccak256_leaves_base_row_major_row_pair_range: keccak
                .load_function("keccak256_leaves_base_row_major_row_pair_range")?,
            keccak256_leaves_base_batched: keccak.load_function("keccak256_leaves_base_batched")?,
            keccak256_leaves_base_row_pair_batched: keccak
                .load_function("keccak256_leaves_base_row_pair_batched")?,
            keccak256_leaves_ext3_batched: keccak.load_function("keccak256_leaves_ext3_batched")?,
            keccak_comp_poly_leaves_ext3: keccak.load_function("keccak_comp_poly_leaves_ext3")?,
            keccak_fri_leaves_ext3: keccak.load_function("keccak_fri_leaves_ext3")?,
            keccak_merkle_level: keccak.load_function("keccak_merkle_level")?,
            merkle_gather_paths: keccak.load_function("merkle_gather_paths")?,
            barycentric_base_batched: bary.load_function("barycentric_base_batched")?,
            barycentric_ext3_batched: bary.load_function("barycentric_ext3_batched")?,
            barycentric_base_batched_strided: bary
                .load_function("barycentric_base_batched_strided")?,
            barycentric_ext3_batched_strided: bary
                .load_function("barycentric_ext3_batched_strided")?,
            gather_rows_base: bary.load_function("gather_rows_base")?,
            gather_rows_ext3: bary.load_function("gather_rows_ext3")?,
            deep_composition_ext3_row: deep.load_function("deep_composition_ext3_row")?,
            bit_reverse_ext3_kernel: deep.load_function("bit_reverse_ext3_interleaved")?,
            fri_fold_ext3: fri.load_function("fri_fold_ext3")?,
            fri_update_twiddles: fri.load_function("fri_update_twiddles")?,
            compute_denoms_ext3: inverse.load_function("compute_denoms_ext3")?,
            block_inclusive_scan_fwd_ext3: inverse
                .load_function("block_inclusive_scan_fwd_ext3")?,
            apply_block_offsets_fwd_ext3: inverse.load_function("apply_block_offsets_fwd_ext3")?,
            block_inclusive_scan_rev_ext3: inverse
                .load_function("block_inclusive_scan_rev_ext3")?,
            apply_block_offsets_rev_ext3: inverse.load_function("apply_block_offsets_rev_ext3")?,
            batch_inverse_combine_ext3: inverse.load_function("batch_inverse_combine_ext3")?,
            logup_fingerprint_ext3: logup.load_function("logup_fingerprint_ext3")?,
            logup_term_ext3: logup.load_function("logup_term_ext3")?,
            logup_row_sum_ext3: logup.load_function("logup_row_sum_ext3")?,
            logup_scan_block_add_ext3: logup.load_function("logup_scan_block_add_ext3")?,
            logup_apply_offsets_add_ext3: logup.load_function("logup_apply_offsets_add_ext3")?,
            logup_finalize_accum_ext3: logup.load_function("logup_finalize_accum_ext3")?,
            logup_assemble_aux_ext3: logup.load_function("logup_assemble_aux_ext3")?,
            constraint_interp_kernel: constraint_interp
                .load_function("constraint_interp_kernel")?,
            constraint_composition_kernel: constraint_interp
                .load_function("constraint_composition_kernel")?,
            decompose_d2_kernel: constraint_interp.load_function("decompose_d2_ext3")?,
            fwd_twiddles: Mutex::new(vec![None; max_log]),
            inv_twiddles: Mutex::new(vec![None; max_log]),
            ctx,
            streams,
            pinned_staging,
            pinned_hashes,
            event_pool,
            staging_hint_u64s: staging_hint,
            hashes_hint_u64s: hashes_hint,
            util_stream,
            next: AtomicUsize::new(0),
            vram_budget_bytes,
        })
    }

    /// VRAM budget in bytes for table-session admission control. `u64::MAX`
    /// when budgeting is disabled (query failed). See the field docs.
    pub fn vram_budget_bytes(&self) -> u64 {
        self.vram_budget_bytes
    }

    /// Publish first-allocation size hints (u64 elems) for the pinned staging
    /// pools, so each worker slot allocates its slab once at final size
    /// instead of climbing a realloc ladder (each pinned alloc/free step costs
    /// ~50ms+). Call once per prove, as soon as table sizes are known; only
    /// raises (never shrinks) the current hints, and only affects slots that
    /// have not allocated yet.
    pub fn set_staging_size_hints(&self, lde_u64s: usize, hashes_u64s: usize) {
        self.staging_hint_u64s
            .fetch_max(lde_u64s, Ordering::Relaxed);
        self.hashes_hint_u64s
            .fetch_max(hashes_u64s, Ordering::Relaxed);
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
/// touch the slab while the DMA is in flight.
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
    // cuEventCreate/Destroy convoys the driver lock under load).
    staging.record_event(stream)?;
    Ok(PendingD2H { staging, n_bytes })
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
