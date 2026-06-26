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

const ARITH_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/arith.ptx"));
const NTT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ntt.ptx"));
const KECCAK_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/keccak.ptx"));
const BARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/barycentric.ptx"));
const DEEP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/deep.ptx"));
const FRI_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fri.ptx"));
const INVERSE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/inverse.ptx"));
const TRACE_CPU_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/trace_cpu.ptx"));
const TRACE_LT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/trace_lt.ptx"));
const TRACE_ALU_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/trace_alu.ptx"));
const TRACE_SHIFT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/trace_shift.ptx"));
const TRACE_MULREM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/trace_mulrem.ptx"));

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

    // arith.ptx
    pub vector_add_u64: CudaFunction,
    pub gl_add: CudaFunction,
    pub gl_sub: CudaFunction,
    pub gl_mul: CudaFunction,
    pub gl_neg: CudaFunction,
    pub ext3_mul: CudaFunction,
    pub ext3_add: CudaFunction,
    pub ext3_sub: CudaFunction,

    // ntt.ptx
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

    // keccak.ptx
    pub keccak256_leaves_base_batched: CudaFunction,
    pub keccak256_leaves_ext3_batched: CudaFunction,
    pub keccak_comp_poly_leaves_ext3: CudaFunction,
    pub keccak_fri_leaves_ext3: CudaFunction,
    pub keccak_merkle_level: CudaFunction,

    // barycentric.ptx
    pub barycentric_base_batched: CudaFunction,
    pub barycentric_ext3_batched: CudaFunction,
    pub barycentric_base_batched_strided: CudaFunction,
    pub barycentric_ext3_batched_strided: CudaFunction,

    // deep.ptx
    pub deep_composition_ext3_row: CudaFunction,

    // fri.ptx
    pub fri_fold_ext3: CudaFunction,
    pub fri_update_twiddles: CudaFunction,

    // inverse.ptx
    pub compute_denoms_ext3: CudaFunction,
    pub block_inclusive_scan_fwd_ext3: CudaFunction,
    pub apply_block_offsets_fwd_ext3: CudaFunction,
    pub block_inclusive_scan_rev_ext3: CudaFunction,
    pub apply_block_offsets_rev_ext3: CudaFunction,
    pub batch_inverse_combine_ext3: CudaFunction,

    // VM trace generation.
    pub trace_cpu_kernel: CudaFunction,
    pub trace_lt_kernel: CudaFunction,
    pub trace_eq_kernel: CudaFunction,
    pub trace_bytewise_kernel: CudaFunction,
    pub trace_shift_kernel: CudaFunction,
    pub trace_mul_kernel: CudaFunction,
    pub trace_dvrm_kernel: CudaFunction,

    // Twiddle caches keyed by log_n.
    fwd_twiddles: Mutex<Vec<Option<Arc<CudaSlice<u64>>>>>,
    inv_twiddles: Mutex<Vec<Option<Arc<CudaSlice<u64>>>>>,
}

impl Backend {
    fn init() -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        // cudarc's default per-slice CudaEvent tracking adds two driver calls
        // per alloc and serialises under the context lock. We never share
        // slices across streams (every call scopes its own buffers and syncs
        // before returning), so the tracking is pure overhead. Disable it.
        unsafe { ctx.disable_event_tracking() };

        let arith = ctx.load_module(Ptx::from_src(ARITH_PTX))?;
        let ntt = ctx.load_module(Ptx::from_src(NTT_PTX))?;
        let keccak = ctx.load_module(Ptx::from_src(KECCAK_PTX))?;
        let bary = ctx.load_module(Ptx::from_src(BARY_PTX))?;
        let deep = ctx.load_module(Ptx::from_src(DEEP_PTX))?;
        let fri = ctx.load_module(Ptx::from_src(FRI_PTX))?;
        let inverse = ctx.load_module(Ptx::from_src(INVERSE_PTX))?;
        let trace_cpu = ctx.load_module(Ptx::from_src(TRACE_CPU_PTX))?;
        let trace_lt = ctx.load_module(Ptx::from_src(TRACE_LT_PTX))?;
        let trace_alu = ctx.load_module(Ptx::from_src(TRACE_ALU_PTX))?;
        let trace_shift = ctx.load_module(Ptx::from_src(TRACE_SHIFT_PTX))?;
        let trace_mulrem = ctx.load_module(Ptx::from_src(TRACE_MULREM_PTX))?;

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
            keccak256_leaves_base_batched: keccak.load_function("keccak256_leaves_base_batched")?,
            keccak256_leaves_ext3_batched: keccak.load_function("keccak256_leaves_ext3_batched")?,
            keccak_comp_poly_leaves_ext3: keccak.load_function("keccak_comp_poly_leaves_ext3")?,
            keccak_fri_leaves_ext3: keccak.load_function("keccak_fri_leaves_ext3")?,
            keccak_merkle_level: keccak.load_function("keccak_merkle_level")?,
            barycentric_base_batched: bary.load_function("barycentric_base_batched")?,
            barycentric_ext3_batched: bary.load_function("barycentric_ext3_batched")?,
            barycentric_base_batched_strided: bary
                .load_function("barycentric_base_batched_strided")?,
            barycentric_ext3_batched_strided: bary
                .load_function("barycentric_ext3_batched_strided")?,
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
            trace_cpu_kernel: trace_cpu.load_function("trace_cpu_kernel")?,
            trace_lt_kernel: trace_lt.load_function("trace_lt_kernel")?,
            trace_eq_kernel: trace_alu.load_function("trace_eq_kernel")?,
            trace_bytewise_kernel: trace_alu.load_function("trace_bytewise_kernel")?,
            trace_shift_kernel: trace_shift.load_function("trace_shift_kernel")?,
            trace_mul_kernel: trace_mulrem.load_function("trace_mul_kernel")?,
            trace_dvrm_kernel: trace_mulrem.load_function("trace_dvrm_kernel")?,
            fwd_twiddles: Mutex::new(vec![None; max_log]),
            inv_twiddles: Mutex::new(vec![None; max_log]),
            ctx,
            streams,
            pinned_staging,
            pinned_hashes,
            util_stream,
            next: AtomicUsize::new(0),
        })
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
    let b = Backend::init()?;
    let _ = BACKEND.set(b);
    Ok(BACKEND.get().expect("backend just initialised"))
}
