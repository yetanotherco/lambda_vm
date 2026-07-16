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
}

// SAFETY: the raw pointer aliases host memory allocated via cuMemHostAlloc.
// We guard concurrent access with a Mutex; the pointer is valid for the
// lifetime of this struct and is freed on drop.
unsafe impl Send for PinnedStaging {}
unsafe impl Sync for PinnedStaging {}

impl PinnedStaging {
    const fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            capacity_elems: 0,
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
    util_stream: Arc<CudaStream>,
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
        // per alloc and serialises under the context lock. Slices are only
        // shared across streams after the producing stream has been host-
        // synchronised (e.g. the retained trace snapshot and the resident
        // LogUp aux buffer; every producer syncs before its handle escapes),
        // so the tracking is pure overhead. Disable it.
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
        let pinned_staging: Vec<Mutex<PinnedStaging>> = (0..n_slots)
            .map(|_| Mutex::new(PinnedStaging::empty()))
            .collect();
        let pinned_hashes: Vec<Mutex<PinnedStaging>> = (0..n_slots)
            .map(|_| Mutex::new(PinnedStaging::empty()))
            .collect();
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
            fwd_twiddles: Mutex::new(vec![None; max_log]),
            inv_twiddles: Mutex::new(vec![None; max_log]),
            ctx,
            streams,
            pinned_staging,
            pinned_hashes,
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
