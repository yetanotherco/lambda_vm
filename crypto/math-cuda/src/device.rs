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

/// Host bytes the pinned staging buffers hold ([`PinnedStaging`]), process
/// wide: page-locked, so they count in the process's resident set but in no
/// allocator's statistics.
static PINNED_HOST_BYTES: AtomicUsize = AtomicUsize::new(0);

/// [`PINNED_HOST_BYTES`]: what the pinned staging buffers hold now.
pub fn pinned_host_bytes() -> usize {
    PINNED_HOST_BYTES.load(Ordering::Relaxed)
}

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
            PINNED_HOST_BYTES.fetch_sub(
                self.capacity_elems * std::mem::size_of::<u64>(),
                Ordering::Relaxed,
            );
            self.ptr = std::ptr::null_mut();
            self.capacity_elems = 0;
        }
        let new_cap = min_elems.next_power_of_two().max(1 << 20); // at least 8 MB
        let bytes = new_cap * std::mem::size_of::<u64>();
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(bytes, 0 /* flags: non-WC */)?
        } as *mut u64;
        PINNED_HOST_BYTES.fetch_add(bytes, Ordering::Relaxed);
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
            PINNED_HOST_BYTES.fetch_sub(
                self.capacity_elems * std::mem::size_of::<u64>(),
                Ordering::Relaxed,
            );
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
const NTT_CM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ntt_cm.cubin"));
const KECCAK_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/keccak.cubin"));
// `RPX_V0_CUBIN` and `RPX_LIMB_CUBIN`: `rpx.cu` at permutation variant 0 and at
// `rpx_paths::LIMB_PERMUTE_VARIANT`, one of which the backend loads (see
// `rpx_paths::limb_permute`). Generated by build.rs.
include!(concat!(env!("OUT_DIR"), "/rpx_cubins.rs"));
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
    /// The stream pool, and the background pool beside it unless
    /// `LAMBDA_VM_STREAM_CLASSES=off` ([`crate::stream_class`]).
    streams: crate::stream_class::StreamPools<Arc<CudaStream>>,
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
    /// Fixed-size double-buffered pinned pairs, one per concurrent
    /// staged transfer. See [`htod_staged`].
    staging_pairs: StagingPool,
    /// Free-list of pre-created events for [`Backend::take_event`].
    event_pool: Mutex<Vec<cudarc::driver::CudaEvent>>,
    /// VRAM budget (bytes) for admission control. See
    /// [`detect_vram_budget_bytes`]; a caller may move it for a phase of its
    /// own ([`Backend::set_vram_budget_bytes`]).
    vram_budget_bytes: AtomicU64,
    /// Device bytes promised to structures that are still alive. See
    /// [`Backend::reserve`].
    reserved: AtomicU64,
    /// The ledger's whole-run and window peaks, this card's own.
    peaks: LedgerPeaks,

    // arith.cubin
    pub vector_add_u64: CudaFunction,
    pub widen_narrow_row_major: CudaFunction,
    pub widen_narrow_col_major: CudaFunction,
    pub column_max_col_major: CudaFunction,
    pub pack_col_major: CudaFunction,
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
    pub mobius_wide_low: CudaFunction,
    pub mobius_wide_tile: CudaFunction,
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

    // ntt_cm.cubin — the column-major LDE engine's passes, indexed by
    // `k - 4` for a pass of k = 4..=8 levels (`crate::lde_cm`), and the WHIR
    // encoding's wide DIT passes by `k - 9` for k = 9, 10 (`lde_cm::plan_wide`).
    pub ntt_cm_dit: [CudaFunction; 5],
    pub ntt_cm_dif: [CudaFunction; 5],
    pub ntt_cm_dit_wide: [CudaFunction; 2],

    // keccak.cubin
    pub keccak256_leaves_base_row_major_row_pair: CudaFunction,
    pub keccak256_leaves_base_row_major_row_pair_range: CudaFunction,
    /// S2 one-row leaves (`rows_per_leaf = 1`): row `reverse_index(i)`, a column range.
    pub keccak256_leaves_base_row_major_row_range: CudaFunction,
    pub keccak256_leaves_base_batched: CudaFunction,
    pub keccak256_leaves_base_coset: CudaFunction,
    pub keccak256_leaves_ext3_coset: CudaFunction,
    pub keccak256_leaves_base_row_pair_batched: CudaFunction,
    pub keccak256_leaves_ext3_batched: CudaFunction,
    pub grind_search: CudaFunction,
    pub keccak_comp_poly_leaves_ext3: CudaFunction,
    pub keccak_fri_leaves_ext3: CudaFunction,
    /// S3 group-leaf FRI layers: `group` consecutive ext3 values per leaf.
    pub keccak_fri_group_leaves_ext3: CudaFunction,
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
    /// S2 one-row leaves (`rows_per_leaf = 1`): row `reverse_index(i)`, a column range.
    pub blake3_leaves_base_row_major_row_range: CudaFunction,
    pub blake3_leaves_base_batched: CudaFunction,
    pub blake3_leaves_base_row_pair_batched: CudaFunction,
    pub blake3_leaves_ext3_batched: CudaFunction,
    pub blake3_comp_poly_leaves_ext3: CudaFunction,
    pub blake3_fri_leaves_ext3: CudaFunction,
    /// S3 group-leaf FRI layers: `group` consecutive ext3 values per leaf.
    pub blake3_fri_group_leaves_ext3: CudaFunction,
    pub blake3_merkle_level: CudaFunction,
    pub blake3_merkle_tail: CudaFunction,
    pub blake3_compress_probe_6r: CudaFunction,
    pub blake3_compress_probe_7r: CudaFunction,
    pub blake3_compress_probe_default: CudaFunction,
    pub blake3_rounds_probe: CudaFunction,
    pub blake3_serialize_felts_probe: CudaFunction,
    pub blake3_blocks_of_felts_probe: CudaFunction,
    pub blake3_chain_probe: CudaFunction,

    // The RPX module — the RPX256 (XHash12) leaf kernels, Merkle level/tail
    // compressors and the permutation probe (see `kernels/rpx.cu`). Twin for
    // twin with the blake3 set above and in the same order; the probe is the
    // only host-visible handle on the bare device permutation, which the parity
    // tests check against the host `Rpx256`.
    pub rpx_leaves_base_row_major_row_pair: CudaFunction,
    pub rpx_leaves_base_row_major_row_pair_range: CudaFunction,
    /// S2 one-row leaves (`rows_per_leaf = 1`): row `reverse_index(i)`, a column range.
    pub rpx_leaves_base_row_major_row_range: CudaFunction,
    pub rpx_leaves_base_batched: CudaFunction,
    pub rpx_leaves_base_row_pair_batched: CudaFunction,
    pub rpx_leaves_ext3_batched: CudaFunction,
    pub rpx_comp_poly_leaves_ext3: CudaFunction,
    pub rpx_fri_leaves_ext3: CudaFunction,
    /// S3 group-leaf FRI layers: `group` consecutive ext3 values per leaf.
    pub rpx_fri_group_leaves_ext3: CudaFunction,
    pub rpx_merkle_level: CudaFunction,
    pub rpx_merkle_tail: CudaFunction,
    pub rpx_permute_probe: CudaFunction,
    pub rpx_grind_search: CudaFunction,
    /// ⛔ DIAGNOSTIC ONLY — the grind search with its executed-permutation
    /// counters. Nothing on a proving path launches it; its one caller is
    /// [`crate::grinding::search_counted`], which reads whether a slow launch
    /// does MORE work or the same work more slowly.
    pub rpx_grind_search_counted: CudaFunction,
    /// The grind with nonces claimed from a work queue (the default RPX grind,
    /// `rpx_paths::grind_queue`), and its counted twin (diagnostic, the same
    /// counters as `rpx_grind_search_counted`).
    pub rpx_grind_search_queue: CudaFunction,
    pub rpx_grind_search_queue_counted: CudaFunction,
    /// A Merkle level and the tail with one permutation per half-warp (the
    /// default Merkle tops, `rpx_paths::warp_merkle`), and the half-warp
    /// permutation's parity probe.
    pub rpx_merkle_level_warp: CudaFunction,
    pub rpx_merkle_tail_warp: CudaFunction,
    pub rpx_permute_warp_probe: CudaFunction,
    /// Permutation-variant probes: the limb primitives, and the permutation
    /// chained at each variant the source instantiates, as `(variant, kernel)`.
    pub rpx_limb_probe: CudaFunction,
    pub rpx_permute_chain_probes: Vec<(u32, CudaFunction)>,

    // The RPX module — the algebraic hash's twins of the keccak entries above.
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
    /// `deep_composition_ext3_hoisted_m{M}` at index `M - 1`, the DEEP row
    /// with its own `M = 1 + num_eval_points` denominators inverted in
    /// registers and its row-independent part hoisted onto the host.
    pub deep_composition_ext3_hoisted: [CudaFunction; 4],

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
    /// `invert_denoms_rowwise_ext3_k{K}` at index `K - 1`.
    pub invert_denoms_rowwise_ext3: [CudaFunction; 8],
    pub logup_fingerprint_ext3: CudaFunction,
    pub logup_term_ext3: CudaFunction,
    pub logup_row_sum_ext3: CudaFunction,
    pub logup_scan_block_add_ext3: CudaFunction,
    pub logup_apply_offsets_add_ext3: CudaFunction,
    pub logup_finalize_accum_ext3: CudaFunction,
    pub logup_assemble_aux_ext3: CudaFunction,

    // sumcheck.cubin
    pub sumcheck_round_ext3: CudaFunction,
    /// D-ARGUE S1-1: the round with integer interpolation nodes.
    pub sumcheck_round_ext3_int: CudaFunction,
    pub sum_partials_ext3: CudaFunction,
    // D-ARGUE stage 1's fused zerocheck (`crate::argue_fused`).
    pub zc_grid01: CudaFunction,
    pub zc_bus_u: CudaFunction,
    pub zc_bus_u_store: CudaFunction,
    pub zc_fold2: CudaFunction,
    pub zc_bus_column: CudaFunction,
    pub zc_halve: CudaFunction,
    pub zc_round_gruen: CudaFunction,
    // D-ARGUE S1-3's GKR layer rounds (`crate::gkr::GruenLayer`).
    pub gkr_eq_levels_ext3: CudaFunction,
    pub gkr_round_gruen: CudaFunction,
    pub gkr_gruen_finish: CudaFunction,
    // The same over an input layer kept without its padding
    // (`crate::gkr::DeviceFractionTree::from_truncated_input`).
    pub gkr_round_gruen_real: CudaFunction,
    pub gkr_gruen_finish_real: CudaFunction,
    // The GKR round kernels whose last block reduces the sums into a slot
    // (`crate::round_slot`).
    pub gkr_round_gruen_slot: CudaFunction,
    pub gkr_round_gruen_real_slot: CudaFunction,
    // A GKR step over two tree layers (`crate::gkr::Gruen2Layer`): the round
    // (whole layer, kept input layer, each with and without a slot), its
    // `H(0)` pass and its finish.
    pub gkr2_round: CudaFunction,
    pub gkr2_round_real: CudaFunction,
    pub gkr2_round_slot: CudaFunction,
    pub gkr2_round_real_slot: CudaFunction,
    pub gkr2_round_h0: CudaFunction,
    pub gkr2_round_h0_real: CudaFunction,
    pub gkr2_finish: CudaFunction,
    pub gkr2_finish_real: CudaFunction,
    // D-BATCH M1-2's input layer from the base columns (`crate::gkr::input_from_columns`).
    pub gkr_input_from_columns: CudaFunction,
    pub sumcheck_fold_ext3: CudaFunction,
    pub mle_fold_base_ext3: CudaFunction,
    pub eq_expand_level_ext3: CudaFunction,
    pub eq_seed_shares_ext3: CudaFunction,
    pub eq_expand_level_shares_ext3: CudaFunction,
    pub program_map_ext3: CudaFunction,
    pub factors_from_columns_ext3: CudaFunction,
    pub mle_lift_base_ext3: CudaFunction,
    pub add_scaled_ext3: CudaFunction,
    pub add_eq_tensor_ext3: CudaFunction,
    pub add_eq_tensor_base_ext3: CudaFunction,
    pub fill_ext3: CudaFunction,
    pub fraction_fold_ext3: CudaFunction,
    pub fraction_fold2_ext3: CudaFunction,
    pub gkr_input_fold2_from_columns: CudaFunction,
    pub fraction_fold_padded_ext3: CudaFunction,
    pub mle_fold_base_ext3_many: CudaFunction,
    pub eq_dot_base_ext3_many: CudaFunction,
    pub batched_column_ext3: CudaFunction,
    pub gather_factor_heads_ext3: CudaFunction,

    // whir_fold.cubin
    pub whir_fold_base_ext3: CudaFunction,
    pub whir_fold_ext3: CudaFunction,
    pub gather_cosets: CudaFunction,
    pub gather_cosets_strided: CudaFunction,
    pub whir_fold_k_base_ext3: CudaFunction,
    pub whir_fold_k_ext3: CudaFunction,
    pub whir_lean_colmap: CudaFunction,
    pub whir_lean_round: CudaFunction,
    pub whir_lean_materialize: CudaFunction,
    /// The lean rounds in tile form (`whir_open::TileSpec`): each column of
    /// tiles' slots and weights, the one pass over the message (a thread a
    /// tile, or a block a tile), a round over the slot tables, their fold,
    /// the bound tables written from them, and that write compared with the
    /// share form's.
    pub whir_tile_plan: CudaFunction,
    pub whir_tile_contract: CudaFunction,
    pub whir_tile_contract_wide: CudaFunction,
    pub whir_tile_round: CudaFunction,
    pub whir_tile_fold: CudaFunction,
    pub whir_tile_materialize: CudaFunction,
    pub whir_tile_materialize_check: CudaFunction,
    /// An interleaved ext3 array's limb as a plane, and back (a successor
    /// encoded from its bound message, `whir::encode_successor`).
    pub ext3_limb_gather: CudaFunction,
    pub ext3_limb_scatter: CudaFunction,

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
    /// The backend whose ledger holds the bytes ([`BackendSlot`]): a
    /// reservation grows, shrinks and is released there, whichever thread
    /// (in whichever role) does it.
    slot: BackendSlot,
}

impl DeviceReservation {
    /// The backend slot whose ledger holds these bytes ([`BackendSlot::index`]).
    pub fn slot_index(&self) -> usize {
        self.slot.index()
    }

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
        let Ok(be) = self.slot.backend() else {
            return false;
        };
        let mut held = be.reserved.load(Ordering::Relaxed);
        loop {
            if held.saturating_add(extra) > be.vram_budget_bytes() {
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
                    be.peaks.note(held + extra);
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
        if let Ok(be) = self.slot.backend() {
            be.reserved.fetch_sub(given, Ordering::Relaxed);
        }
    }
}

impl Drop for DeviceReservation {
    fn drop(&mut self) {
        if let Ok(be) = self.slot.backend() {
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
    drain()?;
    trim_default_mempool();
    Ok(())
}

/// [`drain_and_trim`] without the trim: every stream of the context waited
/// for, so the pool's counts are settled, while the pool keeps the pages it
/// holds for the next allocations. Counted with the trims ([`trims`]) as a
/// context-wide wait.
pub fn drain_untrimmed() -> Result<()> {
    drain()
}

fn drain() -> Result<()> {
    TRIMS.fetch_add(1, Ordering::Relaxed);
    let be = backend()?;
    be.ctx.synchronize()?;
    Ok(())
}

/// Calls to [`drain_and_trim`] this process has made. Each one waits for every
/// stream of the context, so work running on two threads at once stalls
/// together on it. Monotone: a reader brackets the span it cares about.
static TRIMS: AtomicU64 = AtomicU64::new(0);

/// [`TRIMS`]: the [`drain_and_trim`] calls so far.
pub fn trims() -> u64 {
    TRIMS.load(Ordering::Relaxed)
}

/// ★ Bytes the default memory pool has handed out and not taken back: `(now,
/// the most at any instant since the last [`reset_pool_high_water`])`.
///
/// Every allocation this crate makes is stream-ordered and comes from that
/// pool — cudarc allocates with `cuMemAllocAsync` whenever the device has
/// pools, which every card this runs on does — so the second number is the
/// PEAK of what the process held. That is what a working-set claim is tested
/// against: a reservation says what was promised, and `cuMemGetInfo` only what
/// is free at the instant it is asked. Drain first ([`drain_and_trim`]): a free
/// counts when its stream reaches it.
pub fn pool_used_bytes() -> Result<(u64, u64)> {
    use cudarc::driver::sys;
    let be = backend()?;
    let read = |attribute| -> Result<u64> {
        let mut value = 0u64;
        // SAFETY: raw driver calls on the backend's own device; the value is a
        // u64 stack slot, the type both attributes are documented to write.
        unsafe {
            let pool = default_mempool(&be.ctx).ok_or(cudarc::driver::DriverError(
                sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
            ))?;
            sys::cuMemPoolGetAttribute(
                pool,
                attribute,
                &mut value as *mut u64 as *mut core::ffi::c_void,
            )
            .result()?;
        }
        Ok(value)
    };
    Ok((
        read(sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
        read(sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
    ))
}

/// The default pool's bytes reserved from the device, `(now, high-water)`:
/// what the pool holds of the card, used or not.
pub fn pool_reserved_bytes() -> Result<(u64, u64)> {
    use cudarc::driver::sys;
    let be = backend()?;
    let read = |attribute| -> Result<u64> {
        let mut value = 0u64;
        // SAFETY: as in `pool_used_bytes`.
        unsafe {
            let pool = default_mempool(&be.ctx).ok_or(cudarc::driver::DriverError(
                sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
            ))?;
            sys::cuMemPoolGetAttribute(
                pool,
                attribute,
                &mut value as *mut u64 as *mut core::ffi::c_void,
            )
            .result()?;
        }
        Ok(value)
    };
    Ok((
        read(sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        read(sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
    ))
}

/// Restarts [`pool_reserved_bytes`]'s high-water mark (as
/// [`reset_pool_high_water`] does the used one's).
pub fn reset_pool_reserved_high_water() -> Result<()> {
    use cudarc::driver::sys;
    let be = backend()?;
    let zero = 0u64;
    // SAFETY: as in `reset_pool_high_water`.
    unsafe {
        let pool = default_mempool(&be.ctx).ok_or(cudarc::driver::DriverError(
            sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
        ))?;
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            &zero as *const u64 as *mut core::ffi::c_void,
        )
        .result()?;
    }
    Ok(())
}

/// Restarts [`pool_used_bytes`]'s high-water mark. The driver only resets it
/// to zero, and the next allocation raises it to what is then in use — so a
/// reader takes the larger of it and the `now` it read before the reset.
pub fn reset_pool_high_water() -> Result<()> {
    use cudarc::driver::sys;
    let be = backend()?;
    let zero = 0u64;
    // SAFETY: as in `pool_used_bytes`; zero is the one value the driver takes
    // for this attribute.
    unsafe {
        let pool = default_mempool(&be.ctx).ok_or(cudarc::driver::DriverError(
            sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
        ))?;
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            &zero as *const u64 as *mut core::ffi::c_void,
        )
        .result()?;
    }
    Ok(())
}

/// Promises `bytes` against the budget for a caller whose structure outlives
/// the type that spends them.
pub fn reserve(bytes: u64) -> Option<DeviceReservation> {
    backend().ok()?.reserve(bytes)
}

/// [`reserve`] that never asks the retention for its spare bytes: a caller
/// beside another proof's work, whose cache it must not take, waits instead.
pub fn reserve_without_eviction(bytes: u64) -> Option<DeviceReservation> {
    backend().ok()?.reserve_with(bytes, false)
}

/// Argue-surface device fallbacks: the reservation refusals in math-cuda's
/// `sumcheck`, `gkr` and `columns` (and the group upload's declined fill), and
/// the GKR tree's whole-tree promise in `multilinear::gpu`, counted where each
/// one's work moves to the host.
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
/// SCOPE, stated precisely, by function (line numbers go stale). NINE sites
/// count here, each where its work moves to the host:
/// - the EIGHT in `crypto/math-cuda/src`: `sumcheck.rs` (×3), `gkr.rs` (×1),
///   `columns.rs` (×4): `DeviceColumns::upload`'s room, and
///   `DeviceColumns::upload_tables`' room, its staging promise for a packed or
///   records table, and its fill checks (a records table whose log the card
///   refused: the card declines the group and its readers take the host's
///   columns);
/// - the GKR tree's whole-tree promise in `multilinear::gpu`'s
///   `input_layer_tree_impl`, on the consume path only: the host then builds
///   the table's factors and fraction tree. `multilinear::gpu::gkr_tree_refusals`
///   counts it apart as well.
///
/// The other argue-side `reserve` calls in `multilinear::gpu` are NOT
/// fallbacks, and must never be counted here:
/// - `input_layer_tree_impl`'s carry (`reserve(eager)`) is SPECULATIVE: its
///   `None` picks the lazy tree, still on the device;
/// - a prefetch's (`input_layer_tree_deferred`) refusal only means no prefetch;
/// - `DeviceColumns::gather`'s room (a public-reading tree's small store) is
///   speculative too: its `None` takes the old path, which counts its own
///   refusal where its work moves to the host;
/// - `DeviceColumns::upload_store`'s room (a retired prepared stack's store)
///   moves nothing to the host: the reader copies twice on the card;
/// - a kept input layer's tree (`truncated_tree`, `LAMBDA_VM_ARGUE_INPUT_REAL`)
///   is also speculative: its `None` builds today's tree instead;
/// - `reserve_room`'s two callers keep the work on the device. `take_turn` opens
///   unpromised and counts its own refusals (`room_turn_refusals`); the group
///   commit's retention room makes the commit transient, so its leaf pass is
///   redone.
///
/// This is NOT the whole device surface: the commit path has its own counters
/// (`multilinear::gpu::host_fallbacks`, `commit_errors`). Putting a wrong site
/// into the very counter meant to end false numbers is the one thing to avoid.
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

/// Record one argue-side device fallback — bumped at each of the nine sites
/// [`DEVICE_FALLBACKS`] enumerates, and nowhere else.
pub fn note_device_fallback() {
    DEVICE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// ★ The high-water mark of `be.reserved` — the PEAK simultaneous device
/// reservation this process reached, updated wherever the total rises
/// ([`Backend::reserve`] and [`DeviceReservation::grow`]).
///
/// This is the reservation quantity argue's `reserve` is checked against, which
/// the raw device trace cannot report: the raw peak includes the never-purge
/// pool's retained blocks and sits above the reservation budget, while THIS is
/// exactly what the budget gates. A control run reads argue's peak reservation
/// demand here; and while the evictable retention holds only spare bytes, this
/// stays below the budget by construction.
///
/// ★ The same peak over a WINDOW a caller opens and closes — one phase of one
/// proof — beside the whole-run one, which it never touches.
///
/// The whole-run number says how close the process came to the budget; it
/// cannot say WHEN. A change that moves one phase's reservations — giving a
/// group's room back across the argument, say — is tested by that phase's peak,
/// epoch by epoch, and a run's single maximum hides every epoch but one.
///
/// Both peaks are a backend's own ([`LedgerPeaks`]): each card's ledger keeps
/// its own, and the functions below read this thread's backend's.
#[derive(Debug, Default)]
pub struct LedgerPeaks {
    /// The whole-run peak.
    reserved: AtomicU64,
    /// The peak since the window opened.
    window: AtomicU64,
}

impl LedgerPeaks {
    /// The ledger just rose to `now`: both peaks keep the larger.
    fn note(&self, now: u64) {
        self.reserved.fetch_max(now, Ordering::Relaxed);
        self.window.fetch_max(now, Ordering::Relaxed);
    }
}

/// The peak simultaneous device reservation this thread's backend reached
/// (0 without one) — see [`LedgerPeaks`].
pub fn reserved_high_water() -> u64 {
    backend().map_or(0, |be| be.peaks.reserved.load(Ordering::Relaxed))
}

/// Zero this thread's backend's reservation high-water. For a test asserting
/// a delta.
pub fn reset_reserved_high_water() {
    if let Ok(be) = backend() {
        be.peaks.reserved.store(0, Ordering::Relaxed);
    }
}

/// Opens a window on this thread's backend ([`LedgerPeaks`]): its peak starts at what is promised
/// NOW, not at zero, because the total only reports when it RISES — a phase
/// that reserved nothing peaked at whatever it inherited, and a window started
/// at zero would read that as nothing held at all.
///
/// One card's, like the whole-run peak: a window is one phase's only while
/// nothing else on that card reserves during it. The WHIR base proves one
/// epoch at a time; a caller that cannot say the same must mark what it reads.
pub fn reset_window_high_water() {
    if let Ok(be) = backend() {
        be.peaks
            .window
            .store(be.reserved_bytes(), Ordering::Relaxed);
    }
}

/// The peak promised on this thread's backend since the last
/// [`reset_window_high_water`].
pub fn window_high_water() -> u64 {
    backend().map_or(0, |be| be.peaks.window.load(Ordering::Relaxed))
}

/// ★ THE RETENTION EVICTOR — the callback the WHIR leaf-layer retention installs
/// so the allocator can reclaim spare retained bytes UNDER PRESSURE, on a real
/// caller's behalf, without `device.rs` depending on `whir.rs`.
///
/// This is what makes the retention a PRIORITY scheme, not a byte race
/// ([[gpu-two-vramgates-overlap]]): it holds only genuinely-spare bytes, and the
/// instant a real caller (argue) cannot get its reservation, the layers are
/// given back. A plain `fn` pointer — Send + Sync, no allocation — installed once
/// via `OnceLock`; it takes a byte TARGET and returns how many it actually freed.
///
/// ⛔ CONTRACT the evictor MUST honour (see `whir::evict_retained_layers`): it
/// frees by dropping retained layers and calling `DeviceReservation::shrink`, and
/// it MUST NOT call `reserve` or `grow` — those would re-enter the allocator that
/// called it and deadlock.
///
/// It takes the asking backend's slot ([`BackendSlot::index`]) and frees only
/// layers whose promise is that slot's: another card's layers give no room on
/// this card.
static EVICTOR: OnceLock<fn(u64, usize) -> u64> = OnceLock::new();

/// Install the retention evictor. Idempotent — only the first install sticks.
pub fn set_retention_evictor(evictor: fn(u64, usize) -> u64) {
    let _ = EVICTOR.set(evictor);
}

/// Ask the retention to give back at least `target` bytes of `slot`'s budget;
/// returns how many it freed (0 if none is installed or nothing was
/// reclaimable).
fn try_evict_retained(target: u64, slot: usize) -> u64 {
    match EVICTOR.get() {
        Some(evict) => evict(target, slot),
        None => 0,
    }
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
    /// The backend of the card at `ordinal`: its own context (the device's
    /// primary one), stream pools, staging, event pool, twiddle cache and
    /// ledger.
    fn init_on(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
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
        let ntt_cm = ctx.load_module(Ptx::from_binary(NTT_CM_CUBIN.to_vec()))?;
        let keccak = ctx.load_module(Ptx::from_binary(KECCAK_CUBIN.to_vec()))?;
        // The whole RPX module at the permutation `rpx_paths::limb_permute`
        // picks: every RPX kernel of the process runs that variant.
        let rpx_cubin = if crate::rpx_paths::limb_permute() {
            RPX_LIMB_CUBIN
        } else {
            RPX_V0_CUBIN
        };
        let rpx = ctx.load_module(Ptx::from_binary(rpx_cubin.to_vec()))?;
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

        // The stream classes (`LAMBDA_VM_STREAM_CLASSES`): off, every stream as
        // it always was; `priority` (the default), a background pool beside
        // the pool, the pool at the device's greatest priority and the
        // background at its least.
        let classes = crate::stream_class::setting();
        let (foreground_priority, background_priority) = match classes {
            crate::stream_class::StreamClasses::Priority => {
                ctx.bind_to_thread()?;
                let (least, greatest) = cudarc::driver::result::stream::get_priority_range()?;
                (Some(greatest), Some(least))
            }
            crate::stream_class::StreamClasses::Off => (None, None),
        };
        let make_stream = |priority: Option<i32>| match priority {
            Some(p) => ctx.new_stream_with_priority(p),
            None => ctx.new_stream(),
        };
        let mut streams = Vec::with_capacity(STREAM_POOL_SIZE);
        for _ in 0..STREAM_POOL_SIZE {
            streams.push(make_stream(foreground_priority)?);
        }
        let mut background = Vec::new();
        if classes.split() {
            for _ in 0..crate::stream_class::BACKGROUND_STREAMS {
                background.push(make_stream(background_priority)?);
            }
            eprintln!(
                "[gpu] stream classes: {classes:?} · {STREAM_POOL_SIZE} streams at priority {} · {} background streams at priority {}",
                foreground_priority.unwrap_or(0),
                background.len(),
                background_priority.unwrap_or(0),
            );
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
        let util_stream = make_stream(foreground_priority)?;

        // Cache is indexed by log_n. Valid range is [0, TWO_ADICITY] since
        // Goldilocks has roots of unity for orders 2^0..=2^TWO_ADICITY only.
        // Length = TWO_ADICITY + 1 to allow indexing at log_n = TWO_ADICITY.
        let max_log = GoldilocksField::TWO_ADICITY as usize + 1;

        let vram_budget_bytes = detect_vram_budget_bytes(&ctx);

        Ok(Self {
            vector_add_u64: arith.load_function("vector_add_u64")?,
            widen_narrow_row_major: arith.load_function("widen_narrow_row_major")?,
            widen_narrow_col_major: arith.load_function("widen_narrow_col_major")?,
            column_max_col_major: arith.load_function("column_max_col_major")?,
            pack_col_major: arith.load_function("pack_col_major")?,
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
            mobius_wide_low: ntt.load_function("mobius_wide_low")?,
            mobius_wide_tile: ntt.load_function("mobius_wide_tile")?,
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
            ntt_cm_dit: [
                ntt_cm.load_function("ntt_cm_dit_k4")?,
                ntt_cm.load_function("ntt_cm_dit_k5")?,
                ntt_cm.load_function("ntt_cm_dit_k6")?,
                ntt_cm.load_function("ntt_cm_dit_k7")?,
                ntt_cm.load_function("ntt_cm_dit_k8")?,
            ],
            ntt_cm_dif: [
                ntt_cm.load_function("ntt_cm_dif_k4")?,
                ntt_cm.load_function("ntt_cm_dif_k5")?,
                ntt_cm.load_function("ntt_cm_dif_k6")?,
                ntt_cm.load_function("ntt_cm_dif_k7")?,
                ntt_cm.load_function("ntt_cm_dif_k8")?,
            ],
            ntt_cm_dit_wide: [
                ntt_cm.load_function("ntt_cm_dit_k9")?,
                ntt_cm.load_function("ntt_cm_dit_k10")?,
            ],
            keccak256_leaves_base_row_major_row_pair: keccak
                .load_function("keccak256_leaves_base_row_major_row_pair")?,
            keccak256_leaves_base_row_major_row_pair_range: keccak
                .load_function("keccak256_leaves_base_row_major_row_pair_range")?,
            keccak256_leaves_base_row_major_row_range: keccak
                .load_function("keccak256_leaves_base_row_major_row_range")?,
            keccak256_leaves_base_batched: keccak.load_function("keccak256_leaves_base_batched")?,
            keccak256_leaves_base_coset: keccak.load_function("keccak256_leaves_base_coset")?,
            keccak256_leaves_ext3_coset: keccak.load_function("keccak256_leaves_ext3_coset")?,
            keccak256_leaves_base_row_pair_batched: keccak
                .load_function("keccak256_leaves_base_row_pair_batched")?,
            keccak256_leaves_ext3_batched: keccak.load_function("keccak256_leaves_ext3_batched")?,
            grind_search: keccak.load_function("grind_search")?,
            keccak_comp_poly_leaves_ext3: keccak.load_function("keccak_comp_poly_leaves_ext3")?,
            keccak_fri_leaves_ext3: keccak.load_function("keccak_fri_leaves_ext3")?,
            keccak_fri_group_leaves_ext3: keccak.load_function("keccak_fri_group_leaves_ext3")?,
            keccak_merkle_level: keccak.load_function("keccak_merkle_level")?,
            keccak_merkle_tail: keccak.load_function("keccak_merkle_tail")?,
            merkle_gather_paths: keccak.load_function("merkle_gather_paths")?,
            blake3_leaves_base_row_major_row_pair: blake3
                .load_function("blake3_leaves_base_row_major_row_pair")?,
            blake3_leaves_base_row_major_row_pair_range: blake3
                .load_function("blake3_leaves_base_row_major_row_pair_range")?,
            blake3_leaves_base_row_major_row_range: blake3
                .load_function("blake3_leaves_base_row_major_row_range")?,
            blake3_leaves_base_batched: blake3.load_function("blake3_leaves_base_batched")?,
            blake3_leaves_base_row_pair_batched: blake3
                .load_function("blake3_leaves_base_row_pair_batched")?,
            blake3_leaves_ext3_batched: blake3.load_function("blake3_leaves_ext3_batched")?,
            blake3_comp_poly_leaves_ext3: blake3.load_function("blake3_comp_poly_leaves_ext3")?,
            blake3_fri_leaves_ext3: blake3.load_function("blake3_fri_leaves_ext3")?,
            blake3_fri_group_leaves_ext3: blake3.load_function("blake3_fri_group_leaves_ext3")?,
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
            rpx_leaves_base_row_major_row_range: rpx
                .load_function("rpx_leaves_base_row_major_row_range")?,
            rpx_leaves_base_batched: rpx.load_function("rpx_leaves_base_batched")?,
            rpx_leaves_base_row_pair_batched: rpx
                .load_function("rpx_leaves_base_row_pair_batched")?,
            rpx_leaves_ext3_batched: rpx.load_function("rpx_leaves_ext3_batched")?,
            rpx_comp_poly_leaves_ext3: rpx.load_function("rpx_comp_poly_leaves_ext3")?,
            rpx_fri_leaves_ext3: rpx.load_function("rpx_fri_leaves_ext3")?,
            rpx_fri_group_leaves_ext3: rpx.load_function("rpx_fri_group_leaves_ext3")?,
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
            rpx_grind_search_queue: rpx.load_function("rpx_grind_search_queue")?,
            rpx_grind_search_queue_counted: rpx.load_function("rpx_grind_search_queue_counted")?,
            rpx_merkle_level_warp: rpx.load_function("rpx_merkle_level_warp")?,
            rpx_merkle_tail_warp: rpx.load_function("rpx_merkle_tail_warp")?,
            rpx_permute_warp_probe: rpx.load_function("rpx_permute_warp_probe")?,
            rpx_limb_probe: rpx.load_function("rpx_limb_probe")?,
            rpx_permute_chain_probes: [0u32, 1, 3, 4, 5, 7, 9]
                .into_iter()
                .map(|v| {
                    rpx.load_function(&format!("rpx_permute_chain_probe_v{v}"))
                        .map(|f| (v, f))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
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
            deep_composition_ext3_hoisted: [
                deep.load_function("deep_composition_ext3_hoisted_m1")?,
                deep.load_function("deep_composition_ext3_hoisted_m2")?,
                deep.load_function("deep_composition_ext3_hoisted_m3")?,
                deep.load_function("deep_composition_ext3_hoisted_m4")?,
            ],
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
            invert_denoms_rowwise_ext3: [
                inverse.load_function("invert_denoms_rowwise_ext3_k1")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k2")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k3")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k4")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k5")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k6")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k7")?,
                inverse.load_function("invert_denoms_rowwise_ext3_k8")?,
            ],
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
            gather_cosets_strided: whir_fold.load_function("gather_cosets_strided")?,
            whir_fold_k_base_ext3: whir_fold.load_function("whir_fold_k_base_ext3")?,
            whir_fold_k_ext3: whir_fold.load_function("whir_fold_k_ext3")?,
            whir_lean_colmap: whir_fold.load_function("whir_lean_colmap")?,
            whir_lean_round: whir_fold.load_function("whir_lean_round")?,
            whir_lean_materialize: whir_fold.load_function("whir_lean_materialize")?,
            whir_tile_plan: whir_fold.load_function("whir_tile_plan")?,
            whir_tile_contract: whir_fold.load_function("whir_tile_contract")?,
            whir_tile_contract_wide: whir_fold.load_function("whir_tile_contract_wide")?,
            whir_tile_round: whir_fold.load_function("whir_tile_round")?,
            whir_tile_fold: whir_fold.load_function("whir_tile_fold")?,
            whir_tile_materialize: whir_fold.load_function("whir_tile_materialize")?,
            whir_tile_materialize_check: whir_fold.load_function("whir_tile_materialize_check")?,
            ext3_limb_gather: whir_fold.load_function("ext3_limb_gather")?,
            ext3_limb_scatter: whir_fold.load_function("ext3_limb_scatter")?,
            sumcheck_round_ext3: sumcheck.load_function("sumcheck_round_ext3")?,
            sumcheck_round_ext3_int: sumcheck.load_function("sumcheck_round_ext3_int")?,
            sum_partials_ext3: sumcheck.load_function("sum_partials_ext3")?,
            zc_grid01: sumcheck.load_function("zc_grid01")?,
            zc_bus_u: sumcheck.load_function("zc_bus_u")?,
            zc_bus_u_store: sumcheck.load_function("zc_bus_u_store")?,
            zc_fold2: sumcheck.load_function("zc_fold2")?,
            zc_bus_column: sumcheck.load_function("zc_bus_column")?,
            zc_halve: sumcheck.load_function("zc_halve")?,
            zc_round_gruen: sumcheck.load_function("zc_round_gruen")?,
            gkr_eq_levels_ext3: sumcheck.load_function("gkr_eq_levels_ext3")?,
            gkr_round_gruen: sumcheck.load_function("gkr_round_gruen")?,
            gkr_gruen_finish: sumcheck.load_function("gkr_gruen_finish")?,
            gkr_round_gruen_real: sumcheck.load_function("gkr_round_gruen_real")?,
            gkr_gruen_finish_real: sumcheck.load_function("gkr_gruen_finish_real")?,
            gkr_round_gruen_slot: sumcheck.load_function("gkr_round_gruen_slot")?,
            gkr_round_gruen_real_slot: sumcheck.load_function("gkr_round_gruen_real_slot")?,
            gkr2_round: sumcheck.load_function("gkr2_round")?,
            gkr2_round_real: sumcheck.load_function("gkr2_round_real")?,
            gkr2_round_slot: sumcheck.load_function("gkr2_round_slot")?,
            gkr2_round_real_slot: sumcheck.load_function("gkr2_round_real_slot")?,
            gkr2_round_h0: sumcheck.load_function("gkr2_round_h0")?,
            gkr2_round_h0_real: sumcheck.load_function("gkr2_round_h0_real")?,
            gkr2_finish: sumcheck.load_function("gkr2_finish")?,
            gkr2_finish_real: sumcheck.load_function("gkr2_finish_real")?,
            gkr_input_from_columns: sumcheck.load_function("gkr_input_from_columns")?,
            sumcheck_fold_ext3: sumcheck.load_function("sumcheck_fold_ext3")?,
            mle_fold_base_ext3: sumcheck.load_function("mle_fold_base_ext3")?,
            eq_expand_level_ext3: sumcheck.load_function("eq_expand_level_ext3")?,
            eq_seed_shares_ext3: sumcheck.load_function("eq_seed_shares_ext3")?,
            eq_expand_level_shares_ext3: sumcheck.load_function("eq_expand_level_shares_ext3")?,
            program_map_ext3: sumcheck.load_function("program_map_ext3")?,
            factors_from_columns_ext3: sumcheck.load_function("factors_from_columns_ext3")?,
            mle_lift_base_ext3: sumcheck.load_function("mle_lift_base_ext3")?,
            add_scaled_ext3: sumcheck.load_function("add_scaled_ext3")?,
            add_eq_tensor_ext3: sumcheck.load_function("add_eq_tensor_ext3")?,
            add_eq_tensor_base_ext3: sumcheck.load_function("add_eq_tensor_base_ext3")?,
            fill_ext3: sumcheck.load_function("fill_ext3")?,
            fraction_fold_ext3: sumcheck.load_function("fraction_fold_ext3")?,
            fraction_fold2_ext3: sumcheck.load_function("fraction_fold2_ext3")?,
            gkr_input_fold2_from_columns: sumcheck.load_function("gkr_input_fold2_from_columns")?,
            fraction_fold_padded_ext3: sumcheck.load_function("fraction_fold_padded_ext3")?,
            mle_fold_base_ext3_many: sumcheck.load_function("mle_fold_base_ext3_many")?,
            eq_dot_base_ext3_many: sumcheck.load_function("eq_dot_base_ext3_many")?,
            batched_column_ext3: sumcheck.load_function("batched_column_ext3")?,
            gather_factor_heads_ext3: sumcheck.load_function("gather_factor_heads_ext3")?,
            constraint_interp_kernel: constraint_interp
                .load_function("constraint_interp_kernel")?,
            constraint_composition_kernel: constraint_interp
                .load_function("constraint_composition_kernel")?,
            decompose_d2_kernel: constraint_interp.load_function("decompose_d2_ext3")?,
            comp_h_to_slabs_kernel: constraint_interp.load_function("comp_h_to_slabs_ext3")?,
            fwd_twiddles: Mutex::new(vec![None; max_log]),
            inv_twiddles: Mutex::new(vec![None; max_log]),
            ctx,
            streams: crate::stream_class::StreamPools::new(streams, background),
            pinned_staging,
            pinned_hashes,
            staging_pairs: StagingPool::default(),
            event_pool,
            util_stream,
            vram_budget_bytes: AtomicU64::new(vram_budget_bytes),
            reserved: AtomicU64::new(0),
            peaks: LedgerPeaks::default(),
        })
        .and_then(|be: Self| {
            // Set up once per function, so once per card: the wide Möbius
            // tile's dynamic shared memory above the 48 KiB default. A failure
            // fails this backend's start (for the second card: no second
            // backend, its work stays on card 0).
            be.ctx.bind_to_thread()?;
            be.mobius_wide_tile.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                crate::whir::MOBIUS_WIDE_TILE_BYTES as i32,
            )?;
            Ok(be)
        })
    }

    /// VRAM budget in bytes for admission control. `u64::MAX`
    /// when budgeting is disabled (query failed). See the field docs.
    pub fn vram_budget_bytes(&self) -> u64 {
        self.vram_budget_bytes.load(Ordering::Relaxed)
    }

    /// Moves the budget to `bytes` and returns the one it replaces. Promises
    /// already made stay made; only later ones are checked against the new
    /// budget. For a phase that knows its own card room better than the
    /// detected default (the recursion levels after the base), which puts the
    /// old budget back when it ends.
    pub fn set_vram_budget_bytes(&self, bytes: u64) -> u64 {
        self.vram_budget_bytes.swap(bytes, Ordering::Relaxed)
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
        self.reserve_with(bytes, true)
    }

    /// [`Self::reserve`], asking the retention for spare bytes on a miss only
    /// when `evict`.
    fn reserve_with(&self, bytes: u64, evict: bool) -> Option<DeviceReservation> {
        let mut evicted = !evict;
        let mut held = self.reserved.load(Ordering::Relaxed);
        loop {
            if held.saturating_add(bytes) > self.vram_budget_bytes() {
                // ★ Before giving up, ask the retention for spare bytes ONCE. The
                // retention holds only genuinely-spare leaf layers and gives them
                // back here, so a real caller (argue) is never displaced by a
                // cache — the fix for wt16, where the retention won the shared
                // budget and argue fell to the host. Bounded to one pass by
                // `evicted`, so a persistent miss returns None (and the call
                // site's `note_device_fallback` counts it) rather than spinning.
                // A miss the layers cannot cover may keep them
                // (`whir::keep_futile`): nothing is freed, and this returns None
                // as it would after evicting them.
                if !evicted {
                    evicted = true;
                    let deficit = held
                        .saturating_add(bytes)
                        .saturating_sub(self.vram_budget_bytes());
                    if try_evict_retained(deficit, BackendSlot::of(self).index()) > 0 {
                        held = self.reserved.load(Ordering::Relaxed);
                        continue;
                    }
                }
                return None;
            }
            match self.reserved.compare_exchange_weak(
                held,
                held + bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Each card's ledger keeps its own peaks.
                    let slot = BackendSlot::of(self);
                    self.peaks.note(held + bytes);
                    return Some(DeviceReservation {
                        bytes: AtomicU64::new(bytes),
                        slot,
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
    /// streams so their kernel launches overlap on the GPU. A thread marked
    /// background draws from the background pool when one exists
    /// ([`crate::stream_class`]).
    pub fn next_stream(&self) -> Arc<CudaStream> {
        self.streams.next()
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

/// Returns the CUDA backend of this thread's device role, initialising it on
/// first call: card 0's ([`DeviceRole::Main`], every thread unless it says
/// otherwise), or under [`DeviceRole::Openings`] the openings' backend
/// ([`openings_placement`]), which is card 0's own unless the two-card
/// openings are on.
///
/// Returns `Err` when CUDA initialisation fails (no driver, no GPU, PTX load
/// failure). Initialisation is retried on the next call until one succeeds —
/// only a successful `Backend` is cached. One initialisation runs at a time:
/// a caller that finds another initialising waits for it rather than
/// building a second context.
pub fn backend() -> Result<&'static Backend> {
    match DEVICE_ROLE.with(std::cell::Cell::get) {
        DeviceRole::Main => backend_once(),
        DeviceRole::Openings => openings_backend(),
    }
}

// ── Device roles: the openings on a second card ─────────────────────────────

/// What a thread's device work is for, which picks the card [`backend`]
/// returns. Every thread is [`DeviceRole::Main`] unless it entered another
/// role ([`enter_role`], [`set_thread_role`]); a rayon worker of the global
/// pool is never marked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRole {
    /// Everything but phase B's openings: card 0.
    Main,
    /// Phase B's openings: card 1 under the two-card openings
    /// ([`openings_placement`]), card 0 otherwise.
    Openings,
}

thread_local! {
    static DEVICE_ROLE: std::cell::Cell<DeviceRole> = const { std::cell::Cell::new(DeviceRole::Main) };
}

/// This thread's device role.
pub fn current_role() -> DeviceRole {
    DEVICE_ROLE.with(std::cell::Cell::get)
}

/// Restores the thread's previous role when dropped ([`enter_role`]).
#[must_use = "the role lasts while the guard lives"]
pub struct RoleGuard(DeviceRole);

impl Drop for RoleGuard {
    fn drop(&mut self) {
        DEVICE_ROLE.with(|c| c.set(self.0));
    }
}

/// This thread works in `role` until the guard drops.
pub fn enter_role(role: DeviceRole) -> RoleGuard {
    RoleGuard(DEVICE_ROLE.with(|c| c.replace(role)))
}

/// This thread works in `role` for the rest of its life: for a pool's worker
/// threads, set once as each starts.
pub fn set_thread_role(role: DeviceRole) {
    DEVICE_ROLE.with(|c| c.set(role));
}

/// `LAMBDA_VM_TWO_CARD_OPENINGS`: phase B's openings on a second card.
pub const TWO_CARD_OPENINGS_ENV: &str = "LAMBDA_VM_TWO_CARD_OPENINGS";

/// The two-card openings' setting ([`TWO_CARD_OPENINGS_ENV`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoCardOpenings {
    /// `auto`: on when the process sees two cards or more. Not the default
    /// until a two-card box has measured it (bytes and time).
    Auto,
    /// Unset or `0`: off, every role on card 0 (today's). The default.
    Off,
    /// `1`: on. With one card, the openings get a backend of their own on
    /// card 0 (its own streams, staging and ledger, the same context): the
    /// two-card bookkeeping on a one-card box, for tests.
    Forced,
}

impl TwoCardOpenings {
    /// The knob's reading: unset or `0` off (the default), `auto`
    /// [`Self::Auto`], `1` forced; anything else is refused.
    pub fn parse(word: Option<&str>) -> std::result::Result<Self, String> {
        match word.map(str::trim) {
            None | Some("0") => Ok(Self::Off),
            Some("auto") => Ok(Self::Auto),
            Some("1") => Ok(Self::Forced),
            Some(other) => Err(format!("{TWO_CARD_OPENINGS_ENV}={other}: auto, 0 or 1")),
        }
    }
}

/// Where a role's device work runs: the backend slot (0 card 0's, 1 the
/// openings' own) and the card's ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolePlacement {
    pub slot: usize,
    pub ordinal: usize,
}

/// Card 0's backend, every role's without the two-card openings.
pub const MAIN_PLACEMENT: RolePlacement = RolePlacement {
    slot: 0,
    ordinal: 0,
};

/// The placement of `role` under `setting` with `cards` cards visible. Only
/// the openings ever leave slot 0: on card 1 when there are two cards or more
/// (auto or forced), on a second backend of card 0 when forced with one card.
pub fn placement_for(role: DeviceRole, setting: TwoCardOpenings, cards: usize) -> RolePlacement {
    match (role, setting, cards) {
        (DeviceRole::Main, _, _) | (_, TwoCardOpenings::Off, _) | (_, _, 0) => MAIN_PLACEMENT,
        (DeviceRole::Openings, TwoCardOpenings::Auto, 1) => MAIN_PLACEMENT,
        (DeviceRole::Openings, TwoCardOpenings::Forced, 1) => RolePlacement {
            slot: 1,
            ordinal: 0,
        },
        (DeviceRole::Openings, _, _) => RolePlacement {
            slot: 1,
            ordinal: 1,
        },
    }
}

/// Test only: the setting in place of the environment's, process-wide
/// (0 none, 1 auto, 2 off, 3 forced). Set before the openings' backend is made.
static TWO_CARD_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Test only, not for production callers: the two-card setting in place of
/// [`TWO_CARD_OPENINGS_ENV`] (`None`: the environment's). Process-wide, read
/// by [`openings_placement`] on every call; a backend already made for the
/// openings stays made.
#[doc(hidden)]
pub fn set_two_card_openings_for_test(setting: Option<TwoCardOpenings>) {
    let v = match setting {
        None => 0,
        Some(TwoCardOpenings::Auto) => 1,
        Some(TwoCardOpenings::Off) => 2,
        Some(TwoCardOpenings::Forced) => 3,
    };
    TWO_CARD_FORCED.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The process's two-card setting: a test's, else the environment's, read
/// once (an unrecognised word is said once and read as off).
pub fn two_card_openings_setting() -> TwoCardOpenings {
    match TWO_CARD_FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return TwoCardOpenings::Auto,
        2 => return TwoCardOpenings::Off,
        3 => return TwoCardOpenings::Forced,
        _ => {}
    }
    static SETTING: OnceLock<TwoCardOpenings> = OnceLock::new();
    *SETTING.get_or_init(|| {
        TwoCardOpenings::parse(std::env::var(TWO_CARD_OPENINGS_ENV).ok().as_deref()).unwrap_or_else(
            |e| {
                eprintln!("{e} — read as off");
                TwoCardOpenings::Off
            },
        )
    })
}

/// The cards the driver sees, counted once; 0 without a driver (cudarc panics
/// when it cannot load the driver library: caught here and read as none).
pub fn card_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::panic::catch_unwind(CudaContext::device_count)
            .ok()
            .and_then(|n| n.ok())
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    })
}

/// Where the openings' device work runs in this process. Off, the driver is
/// not asked for its cards.
pub fn openings_placement() -> RolePlacement {
    match two_card_openings_setting() {
        TwoCardOpenings::Off => MAIN_PLACEMENT,
        setting => placement_for(DeviceRole::Openings, setting, card_count()),
    }
}

/// Whether the openings run on a backend of their own (card 1, or forced).
pub fn two_card_openings() -> bool {
    openings_placement().slot != 0
}

static SECOND_BACKEND: OnceLock<Backend> = OnceLock::new();

/// Which of the process's backends an object belongs to: card 0's
/// ([`backend_once`]) or the openings' own ([`SECOND_BACKEND`]). What holds
/// a backend's ledger bytes or pooled events keeps its slot, so it gives them
/// back to that backend even when a thread in another role drops it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSlot {
    Main,
    Second,
}

impl BackendSlot {
    /// The slot of `be`.
    pub fn of(be: &Backend) -> Self {
        match SECOND_BACKEND.get() {
            Some(o) if std::ptr::eq(o, be) => Self::Second,
            _ => Self::Main,
        }
    }

    /// The backend of this slot.
    pub fn backend(self) -> Result<&'static Backend> {
        match self {
            Self::Main => backend_once(),
            Self::Second => SECOND_BACKEND.get().map_or_else(backend_once, Ok),
        }
    }
}

impl BackendSlot {
    /// 0 for card 0's backend, 1 for the second.
    pub fn index(self) -> usize {
        match self {
            Self::Main => 0,
            Self::Second => 1,
        }
    }
}

/// A value made once per backend (card 0's, the second's): kernels loaded in
/// a backend's context and tables in its memory belong to that backend
/// alone, so a cache of them is one per backend, never one per process.
/// With one backend it is a single cell, as before.
pub(crate) struct PerBackend<T>([OnceLock<T>; 2]);

impl<T> PerBackend<T> {
    pub(crate) const fn new() -> Self {
        Self([OnceLock::new(), OnceLock::new()])
    }

    /// The cell of `be`.
    pub(crate) fn of(&self, be: &Backend) -> &OnceLock<T> {
        &self.0[BackendSlot::of(be).index()]
    }

    /// The cell of this thread's backend ([`backend`]).
    pub(crate) fn current(&self) -> Result<&OnceLock<T>> {
        Ok(self.of(backend()?))
    }
}

/// The openings' backend: card 0's under [`MAIN_PLACEMENT`], else the second
/// slot's, made on first use. A second backend that fails to start is said
/// once and the openings fall back to card 0's for the call (bytes are the
/// same on either card).
fn openings_backend() -> Result<&'static Backend> {
    let placement = openings_placement();
    if placement.slot == 0 {
        return backend_once();
    }
    // Card 0's first: the pinned pool and the process's first context are
    // card 0's, as without the openings' backend.
    let main = backend_once()?;
    if let Some(b) = SECOND_BACKEND.get() {
        return Ok(b);
    }
    static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if FAILED.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(main);
    }
    static INIT: Mutex<()> = Mutex::new(());
    let _init = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = SECOND_BACKEND.get() {
        return Ok(b);
    }
    match Backend::init_on(placement.ordinal) {
        Ok(b) => {
            eprintln!(
                "[gpu] two-card openings: phase B's openings on card {} ({TWO_CARD_OPENINGS_ENV}={:?}, {} card(s) seen)",
                placement.ordinal,
                two_card_openings_setting(),
                card_count()
            );
            // Two backends on one card (forced, one card): two full ledgers
            // would promise the card twice, so each gets half.
            if placement.ordinal == main.ordinal() {
                let half = main.vram_budget_bytes() / 2;
                main.set_vram_budget_bytes(half);
                b.set_vram_budget_bytes(b.vram_budget_bytes().min(half));
                eprintln!(
                    "[gpu] two backends on card {}: each ledger's budget halved to {:.2} GiB",
                    placement.ordinal,
                    half as f64 / (1u64 << 30) as f64
                );
            }
            let _ = SECOND_BACKEND.set(b);
            match SECOND_BACKEND.get() {
                Some(b) => Ok(b),
                None => Ok(main),
            }
        }
        Err(e) => {
            FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[gpu] two-card openings: card {}'s backend did not start ({e}); the openings stay on card 0",
                placement.ordinal
            );
            Ok(main)
        }
    }
}

/// The openings' own rayon pool, once made ([`on_openings_card`]); `None`
/// inside when it could not be built.
static OPENINGS_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Whether the openings' device work really runs on a backend of their own:
/// the two-card openings on, their backend started (not fallen back to card
/// 0's) and their pool built. A counter of groups opened there reads it after
/// the openings ran, so a fallback to card 0 is not counted.
pub fn openings_on_own_backend() -> bool {
    two_card_openings()
        && SECOND_BACKEND.get().is_some()
        && OPENINGS_POOL.get().is_some_and(Option::is_some)
}

/// Runs `f` as the openings' device work. Without the two-card openings (or
/// without a pool for them) it is `f()` on this thread, as before. With them, `f` runs under
/// [`DeviceRole::Openings`] inside a rayon pool of its own whose workers are
/// all openings threads, so a parallel iterator inside `f` draws the openings'
/// card too (a worker of the global pool is [`DeviceRole::Main`]: card 0).
pub fn on_openings_card<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if !two_card_openings() {
        return f();
    }
    let pool = OPENINGS_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads().max(1))
            .thread_name(|i| format!("openings-card-{i}"))
            .start_handler(|_| {
                set_thread_role(DeviceRole::Openings);
                // Background-class draws, as on the opener's own thread
                // (phase B's openings take the pinned read-backs there), for
                // the worker's life.
                std::mem::forget(crate::stream_class::enter(
                    crate::stream_class::StreamClass::Background,
                ));
            })
            .build()
            .map_err(|e| {
                eprintln!("[gpu] two-card openings: no pool of their own ({e}); on this thread")
            })
            .ok()
    });
    match pool {
        Some(pool) => {
            let _role = enter_role(DeviceRole::Openings);
            pool.install(f)
        }
        // Without a pool of their own the openings stay on card 0 whole: a
        // role on this thread alone would split them over two cards.
        None => f(),
    }
}

/// The ordinal of the card a backend's context is on.
impl Backend {
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }
}

/// Sets the backend up ahead of its first use ([`backend`]), without counting
/// as a device entry: a prove that will reach the card starts it as the
/// prove starts, on a thread of its own, so its first upload does not wait for
/// the context, the modules and the staging pools. A failure is left for the
/// first use to meet.
pub fn set_up_backend() {
    let Ok(main) = backend_once() else { return };
    // Under the two-card openings the second backend is started here too, and
    // the caches its first opening would load (the BLAKE3 WHIR kernels, the
    // root tables), so phase B does not wait for them. Off, the driver is not
    // asked for its cards.
    if two_card_openings() {
        let _role = enter_role(DeviceRole::Openings);
        if let Ok(second) = backend()
            && !std::ptr::eq(second, main)
        {
            let _ = crate::blake3_whir::kernels();
            let _ = crate::lde_cm::root_tables(second);
        }
    }
}

/// The backend once [`backend_once`] has made it.
static BACKEND: OnceLock<Backend> = OnceLock::new();

/// The backend if it is already up, without making it or waiting for it: for
/// a caller that must not hold up its thread on the initialisation (the pinned
/// arena, which starts page-locking once the context exists).
pub fn backend_if_ready() -> Option<&'static Backend> {
    BACKEND.get()
}

/// Card 0's backend by name, whatever this thread's role: for what belongs to
/// the process rather than to a card's work (the pinned pool's page-locking,
/// in-place registrations, the set-up at the prove's start).
pub fn main_backend() -> Result<&'static Backend> {
    backend_once()
}

/// The slot of this thread's backend: 0 under [`DeviceRole::Main`], without a
/// second backend, or without a card; 1 when this thread works for the second.
pub fn current_slot() -> usize {
    match current_role() {
        DeviceRole::Main => 0,
        // This thread's backend first: on a thread's first device call in the
        // second role it is what makes the second backend.
        _ => match backend() {
            Ok(be) => usize::from(BackendSlot::of(be) == BackendSlot::Second),
            Err(_) => 0,
        },
    }
}

/// [`backend`] without the entry count.
fn backend_once() -> Result<&'static Backend> {
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _init = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = match Backend::init_on(0) {
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
            let copy_t0 = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(src.add(byte_off), staging.ptr as *mut u8, this_bytes);
            STAGING_STATS.note_in(false, this_bytes, copy_t0.elapsed());
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

// ── Per-transfer, double-buffered pinned staging ─────────────────────────────

// Per-transfer staging: the row-major commit's trace upload and its
// retained-LDE download each take a staging pair of their own
// ([`htod_staged`], [`dtoh_staged_into`]) rather than the shared per-worker
// slab of [`htod_via`] / [`async_dtoh_via`], which a test's
// [`set_staging_pairs_override`] can still select.
//
// ⛔ WHY PAIRS. The per-table scheduler's driver threads are not rayon workers,
// so [`Backend::worker_slot`] sends every one of them to slot 0. Through the
// shared slab every driver stages through ONE slab, one single-buffered chunk at
// a time, and a driver copying a retained LDE out holds that slab's mutex for
// the whole host copy while the others queue with the card idle. The slab also
// grows to the largest retained LDE's next power of two and never shrinks, which
// is pinned host memory for the rest of the process.
thread_local! {
    /// The calling thread's override of [`staging_pairs_enabled`], so one test
    /// process can run both paths. `None` is the default, pairs.
    static STAGING_PAIRS_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether the calling thread stages the row-major commit's transfers through
/// pairs: its override if set, else yes.
pub fn staging_pairs_enabled() -> bool {
    STAGING_PAIRS_OVERRIDE.with(|o| o.get()).unwrap_or(true)
}

/// Force [`staging_pairs_enabled`] on the calling thread; `None` restores the
/// default.
pub fn set_staging_pairs_override(on: Option<bool>) {
    STAGING_PAIRS_OVERRIDE.with(|o| o.set(on));
}

thread_local! {
    /// The calling thread's [`grid_cap`].
    static GRID_CAP: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// The most blocks a long kernel launched from the calling thread may take,
/// its work walked by a grid-stride loop: `None` (the default) launches the
/// full grid. Set by [`with_grid_cap`] around work that runs beside
/// latency-bound kernels, so that it holds at most that many blocks resident
/// and the others find SMs free. The values are the same at any grid.
pub fn grid_cap() -> Option<u32> {
    GRID_CAP.with(|c| c.get())
}

/// `full` blocks, cut to the calling thread's [`grid_cap`] (at least one).
pub fn capped_grid(full: u32) -> u32 {
    match grid_cap() {
        Some(cap) => full.min(cap.max(1)),
        None => full,
    }
}

/// Runs `f` with the calling thread's [`grid_cap`] set to `cap`, the previous
/// cap restored after (also on a panic), so calls nest.
pub fn with_grid_cap<R>(cap: Option<u32>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<u32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GRID_CAP.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(GRID_CAP.with(|c| c.replace(cap)));
    f()
}

/// Bytes in one staging buffer. A pair holds two.
pub const STAGED_CHUNK_BYTES: usize = 32 << 20;

/// The most pairs a process ever allocates, which bounds the pairs' pinned
/// footprint at `MAX_STAGING_PAIRS × 2 × STAGED_CHUNK_BYTES` = 512 MiB and the
/// number of pinned allocations at sixteen. Pairs are made on demand, lent to
/// one transfer at a time and never freed or grown — the bound whose absence
/// made pre-sized per-worker slabs a measured regression (see the note on
/// pinned-staging size hints in `stark::prover::multi_prove`).
pub const MAX_STAGING_PAIRS: usize = 8;

/// Two fixed-size pinned buffers, each carrying the event of the last DMA that
/// read or wrote it.
struct StagingPair {
    bufs: [PinnedStaging; 2],
}

impl StagingPair {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let mut bufs = [PinnedStaging::empty(), PinnedStaging::empty()];
        for buf in &mut bufs {
            buf.ensure_capacity(STAGED_CHUNK_BYTES / 8, ctx)?;
            buf.event = Some(ctx.new_event(None)?);
        }
        Ok(Self { bufs })
    }
}

/// The pairs not on loan, and how many exist.
#[derive(Default)]
struct StagingPool {
    free: Mutex<Vec<StagingPair>>,
    returned: std::sync::Condvar,
    created: AtomicUsize,
}

impl StagingPool {
    /// A pair for one transfer: a free one, else a new one while fewer than
    /// [`MAX_STAGING_PAIRS`] exist, else the next one handed back. Never nested
    /// (a transfer holds one pair and asks for nothing else while it does), so
    /// the wait cannot deadlock.
    fn lend(&self, ctx: &Arc<CudaContext>) -> Result<PairLoan<'_>> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(pair) = free.pop() {
                return Ok(PairLoan {
                    pool: self,
                    pair: Some(pair),
                });
            }
            if self.created.load(Ordering::Relaxed) < MAX_STAGING_PAIRS {
                // Counted under the lock, so the cap cannot be overshot.
                self.created.fetch_add(1, Ordering::Relaxed);
                drop(free);
                return match StagingPair::new(ctx) {
                    Ok(pair) => Ok(PairLoan {
                        pool: self,
                        pair: Some(pair),
                    }),
                    Err(e) => {
                        self.created.fetch_sub(1, Ordering::Relaxed);
                        Err(e)
                    }
                };
            }
            STAGING_STATS.pair_waits.fetch_add(1, Ordering::Relaxed);
            free = self.returned.wait(free).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Makes pairs until `want` exist (at most [`MAX_STAGING_PAIRS`]) and puts
    /// them in the free list; returns how many this call made. One pair at a
    /// time, the lock not held across a pinned allocation, so a transfer that
    /// lends meanwhile is never kept waiting behind this.
    fn prewarm(&self, ctx: &Arc<CudaContext>, want: usize) -> Result<usize> {
        let want = want.min(MAX_STAGING_PAIRS);
        let mut made = 0;
        loop {
            {
                let _free = self.free.lock().unwrap_or_else(|e| e.into_inner());
                if self.created.load(Ordering::Relaxed) >= want {
                    return Ok(made);
                }
                // Counted under the lock, as `lend` counts, so the cap holds.
                self.created.fetch_add(1, Ordering::Relaxed);
            }
            let pair = match StagingPair::new(ctx) {
                Ok(pair) => pair,
                Err(e) => {
                    self.created.fetch_sub(1, Ordering::Relaxed);
                    return Err(e);
                }
            };
            self.free
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(pair);
            self.returned.notify_one();
            made += 1;
        }
    }
}

/// Makes the staging pairs a later transfer would make on first use, until
/// `want` exist (at most [`MAX_STAGING_PAIRS`]); returns how many this call
/// made. Each pair is two pinned allocations, which take the driver's lock for
/// several milliseconds: a caller that runs this before the transfers that
/// need the pairs takes those allocations out of their windows. The pairs are
/// the ones [`htod_staged`] would make, so nothing a transfer moves changes.
pub fn prewarm_staging_pairs(want: usize) -> Result<usize> {
    let be = backend()?;
    be.ctx.bind_to_thread()?;
    be.staging_pairs.prewarm(&be.ctx, want)
}

/// How many staging pairs exist (made on first use or by
/// [`prewarm_staging_pairs`]).
pub fn staging_pairs_created() -> usize {
    backend().map_or(0, |be| be.staging_pairs.created.load(Ordering::Relaxed))
}

/// A pair on loan to one transfer, handed back when dropped. Every exit path of
/// a transfer leaves no DMA in flight on a buffer without that buffer's event
/// recorded after it, so the next borrower's event waits are enough.
struct PairLoan<'a> {
    pool: &'a StagingPool,
    pair: Option<StagingPair>,
}

impl PairLoan<'_> {
    fn buf(&mut self, k: usize) -> &mut PinnedStaging {
        &mut self.pair.as_mut().expect("present until drop").bufs[k & 1]
    }
}

impl Drop for PairLoan<'_> {
    fn drop(&mut self) {
        if let Some(pair) = self.pair.take() {
            self.pool
                .free
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(pair);
            self.pool.returned.notify_one();
        }
    }
}

/// What the staging surface moved and what its host copies cost, by path —
/// counted on BOTH arms so an A/B reads the same quantities on each. `in` is the
/// commit's trace upload (host memcpy into pinned memory); `out` is the retained
/// LDE's download, timed from the DMA's enqueue to the host holding every value.
struct StagingStats {
    /// `[shared slab, staged]`.
    in_bytes: [AtomicU64; 2],
    in_nanos: [AtomicU64; 2],
    out_bytes: [AtomicU64; 2],
    out_nanos: [AtomicU64; 2],
    pair_waits: AtomicU64,
}

static STAGING_STATS: StagingStats = StagingStats {
    in_bytes: [AtomicU64::new(0), AtomicU64::new(0)],
    in_nanos: [AtomicU64::new(0), AtomicU64::new(0)],
    out_bytes: [AtomicU64::new(0), AtomicU64::new(0)],
    out_nanos: [AtomicU64::new(0), AtomicU64::new(0)],
    pair_waits: AtomicU64::new(0),
};

impl StagingStats {
    fn note_in(&self, staged: bool, bytes: usize, took: std::time::Duration) {
        let i = usize::from(staged);
        self.in_bytes[i].fetch_add(bytes as u64, Ordering::Relaxed);
        self.in_nanos[i].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }

    fn note_out(&self, staged: bool, bytes: usize, took: std::time::Duration) {
        let i = usize::from(staged);
        self.out_bytes[i].fetch_add(bytes as u64, Ordering::Relaxed);
        self.out_nanos[i].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Record one retained-LDE download taken through the shared slab: `bytes`
/// landed in a host `Vec` `took` after the DMA was queued.
pub fn note_shared_slab_download(bytes: usize, took: std::time::Duration) {
    STAGING_STATS.note_out(false, bytes, took);
}

/// The staging counters as numbers, for a test that asserts a path was taken.
#[derive(Clone, Copy, Debug)]
pub struct StagingTotals {
    pub shared_in_bytes: u64,
    pub staged_in_bytes: u64,
    pub shared_out_bytes: u64,
    pub staged_out_bytes: u64,
    /// Staging pairs allocated so far.
    pub pairs: usize,
    /// Times a transfer waited because every pair was on loan.
    pub pair_waits: u64,
}

/// The staging counters since the process started.
pub fn staging_totals() -> StagingTotals {
    let s = &STAGING_STATS;
    StagingTotals {
        shared_in_bytes: s.in_bytes[0].load(Ordering::Relaxed),
        staged_in_bytes: s.in_bytes[1].load(Ordering::Relaxed),
        shared_out_bytes: s.out_bytes[0].load(Ordering::Relaxed),
        staged_out_bytes: s.out_bytes[1].load(Ordering::Relaxed),
        pairs: backend().map_or(0, |be| be.staging_pairs.created.load(Ordering::Relaxed)),
        pair_waits: s.pair_waits.load(Ordering::Relaxed),
    }
}

// How a preprocessed commit brings its precomputed tree's node buffer to the
// host on a tree-cache miss: through a staging pair
// ([`dtoh_staged_uncounted`]) straight into the host tree's node vector, with
// the multiplicity tree queued behind the last chunk. A test's
// [`set_tree_download_override`] can still take the pageable path: one
// pageable copy into a zeroed byte vector, converted to nodes at the end of
// the commit. The nodes are the same bytes either way; only the transfer
// differs. #1013's median 25475471 recursion took 2.9 s (ULTRA) and 6.3 s
// (RYZEN) less staged (I-COPIES §6, §8).
thread_local! {
    /// The calling thread's override of [`tree_download_staged`], so one test
    /// process can run both paths. `None` is the default, staged.
    static TREE_DOWNLOAD_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether the calling thread downloads precomputed trees staged: its override
/// if set, else yes.
pub fn tree_download_staged() -> bool {
    TREE_DOWNLOAD_OVERRIDE.with(|o| o.get()).unwrap_or(true)
}

/// Force [`tree_download_staged`] on the calling thread; `None` restores the
/// default.
pub fn set_tree_download_override(on: Option<bool>) {
    TREE_DOWNLOAD_OVERRIDE.with(|o| o.set(on));
}

/// Precomputed-tree downloads by path (`[pageable, staged]`): trees, bytes, and
/// the time from the tree being built to the host holding every node — the
/// pageable path's zero-fill, copy and conversion to nodes included.
static TREE_DOWNLOADS: [[AtomicU64; 3]; 2] = [
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
];

/// Record one precomputed-tree download of `bytes` that took `took`.
pub fn note_tree_download(staged: bool, bytes: usize, took: std::time::Duration) {
    let c = &TREE_DOWNLOADS[usize::from(staged)];
    c[0].fetch_add(1, Ordering::Relaxed);
    c[1].fetch_add(bytes as u64, Ordering::Relaxed);
    c[2].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
}

/// The precomputed-tree download counters, `[pageable, staged]` each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeDownloadTotals {
    pub trees: [u64; 2],
    pub bytes: [u64; 2],
    pub nanos: [u64; 2],
}

impl TreeDownloadTotals {
    /// What moved between `earlier` and `self`.
    pub fn since(&self, earlier: &Self) -> Self {
        let d = |a: [u64; 2], b: [u64; 2]| [a[0] - b[0], a[1] - b[1]];
        Self {
            trees: d(self.trees, earlier.trees),
            bytes: d(self.bytes, earlier.bytes),
            nanos: d(self.nanos, earlier.nanos),
        }
    }

    /// One line: `pageable <n> trees <GB> in <s> (<GB/s>) | staged …`.
    pub fn line(&self) -> String {
        let one = |i: usize| {
            let gb = self.bytes[i] as f64 / 1e9;
            let s = self.nanos[i] as f64 / 1e9;
            format!(
                "{} {} trees {gb:.2} GB in {s:.2} s ({:.2} GB/s)",
                ["pageable", "staged"][i],
                self.trees[i],
                if s > 0.0 { gb / s } else { 0.0 }
            )
        };
        format!("{} | {}", one(0), one(1))
    }
}

/// The precomputed-tree download counters since the process started.
pub fn tree_download_totals() -> TreeDownloadTotals {
    let load = |k: usize| {
        [
            TREE_DOWNLOADS[0][k].load(Ordering::Relaxed),
            TREE_DOWNLOADS[1][k].load(Ordering::Relaxed),
        ]
    };
    TreeDownloadTotals {
        trees: load(0),
        bytes: load(1),
        nanos: load(2),
    }
}

#[cfg(test)]
mod tree_download_tests {
    use super::{TreeDownloadTotals, set_tree_download_override, tree_download_staged};

    /// The staged download is the default; a thread's override takes either
    /// path and `None` restores the default.
    #[test]
    fn staged_is_the_default() {
        assert!(
            tree_download_staged(),
            "the default must be the staged download"
        );
        set_tree_download_override(Some(false));
        assert!(!tree_download_staged());
        set_tree_download_override(None);
        assert!(tree_download_staged());
    }

    #[test]
    fn totals_subtract_and_print_per_path() {
        let earlier = TreeDownloadTotals {
            trees: [1, 2],
            bytes: [1_000_000_000, 0],
            nanos: [500_000_000, 0],
        };
        let now = TreeDownloadTotals {
            trees: [3, 2],
            bytes: [5_000_000_000, 0],
            nanos: [2_500_000_000, 0],
        };
        let d = now.since(&earlier);
        assert_eq!(d.trees, [2, 0]);
        assert_eq!(
            d.line(),
            "pageable 2 trees 4.00 GB in 2.00 s (2.00 GB/s) | staged 0 trees 0.00 GB in 0.00 s (0.00 GB/s)"
        );
    }
}

/// The shared slabs' pinned footprint, LDE staging and Merkle hashes together:
/// `(bytes, slots allocated, slots busy)`. A slab grows to a power of two and
/// never shrinks, so this is its high-water so far. A slot another thread holds
/// right now is counted as busy, not waited for.
fn shared_slab_footprint(be: &Backend) -> (u64, usize, usize) {
    slab_footprint(be.pinned_staging.iter().chain(&be.pinned_hashes))
}

/// [`shared_slab_footprint`] over any set of slots.
fn slab_footprint<'a>(
    slots: impl Iterator<Item = &'a Mutex<PinnedStaging>>,
) -> (u64, usize, usize) {
    let (mut bytes, mut allocated, mut busy) = (0u64, 0usize, 0usize);
    for slot in slots {
        let elems = match slot.try_lock() {
            Ok(s) => s.capacity_elems,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner().capacity_elems,
            Err(std::sync::TryLockError::WouldBlock) => {
                busy += 1;
                continue;
            }
        };
        if elems > 0 {
            allocated += 1;
            bytes += (elems * std::mem::size_of::<u64>()) as u64;
        }
    }
    (bytes, allocated, busy)
}

/// One line for the log: the staging surface's traffic and host-copy seconds per
/// path since the process started, the staging pairs' pool size and waits, and the
/// shared slabs' pinned footprint (what the retained-LDE downloads grow while
/// the shared-slab path is taken).
pub fn staging_report() -> String {
    staging_line(
        backend().map_or(0, |be| be.staging_pairs.created.load(Ordering::Relaxed)),
        backend().map_or((0, 0, 0), shared_slab_footprint),
    )
}

/// [`staging_report`]'s line from the counters, the pairs allocated and the
/// shared slabs' `(bytes, slots allocated, slots busy)`.
fn staging_line(pairs: usize, (slab_bytes, slabs, busy): (u64, usize, usize)) -> String {
    let s = &STAGING_STATS;
    let gb = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
    let secs = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
    let rate = |b: &AtomicU64, t: &AtomicU64| {
        let t = secs(t);
        if t > 0.0 { gb(b) / t } else { 0.0 }
    };
    format!(
        "staging: upload shared-slab {:.2} GB in {:.2} s memcpy ({:.1} GB/s) · staged {:.2} GB in \
         {:.2} s ({:.1} GB/s) | retained-LDE download shared-slab {:.2} GB in {:.2} s ({:.1} GB/s) · \
         staged {:.2} GB in {:.2} s ({:.1} GB/s) | staging pairs {} of {MAX_STAGING_PAIRS} · waits {} \
         | shared slabs pinned {:.2} GiB in {} slot(s), {} busy",
        gb(&s.in_bytes[0]),
        secs(&s.in_nanos[0]),
        rate(&s.in_bytes[0], &s.in_nanos[0]),
        gb(&s.in_bytes[1]),
        secs(&s.in_nanos[1]),
        rate(&s.in_bytes[1], &s.in_nanos[1]),
        gb(&s.out_bytes[0]),
        secs(&s.out_nanos[0]),
        rate(&s.out_bytes[0], &s.out_nanos[0]),
        gb(&s.out_bytes[1]),
        secs(&s.out_nanos[1]),
        rate(&s.out_bytes[1], &s.out_nanos[1]),
        pairs,
        s.pair_waits.load(Ordering::Relaxed),
        slab_bytes as f64 / (1u64 << 30) as f64,
        slabs,
        busy,
    )
}

/// Host→device copy through a staging pair of its own, double-buffered: the
/// host fills one buffer while the previous chunk's DMA drains the other.
///
/// Returns once the LAST chunk is queued, without waiting for it: every host
/// read of `src_host` is done by then, so it is reusable on return, and work
/// the caller queues on `stream` next is ordered behind the copy. The buffers
/// stay safe because each carries the event of the DMA that last read it, and
/// whoever refills one — this call two chunks on, or the pair's next borrower on
/// any stream — waits on that event first.
///
/// Under a test's [`set_staging_pairs_override`] the upload takes [`htod_via`] instead.
pub fn htod_staged<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    src_host: &[T],
    dst: &mut cudarc::driver::CudaViewMut<'_, T>,
) -> Result<()> {
    use cudarc::driver::DevicePtrMut;
    assert!(
        dst.len() >= src_host.len(),
        "htod_staged: destination shorter than source"
    );
    let elem_size = std::mem::size_of::<T>();
    assert!(
        elem_size <= STAGED_CHUNK_BYTES,
        "htod_staged: an element larger than a staging buffer"
    );
    let n_elems = src_host.len();
    if std::mem::size_of_val(src_host) == 0 {
        return Ok(());
    }
    // A source page-locked in place ([`crate::registered`]) skips the staging
    // copy: the same bytes to the same offsets, as direct DMAs. With nothing
    // registered (the default) this is one atomic load.
    if crate::registered::any()
        && let Some(plan) = crate::registered::direct_plan(
            src_host.as_ptr() as *const u8,
            std::mem::size_of_val(src_host),
        )
    {
        return htod_direct(stream, src_host, dst, &plan);
    }
    // Whole elements per chunk, so a `T` never straddles two buffers.
    let chunk_elems = STAGED_CHUNK_BYTES / elem_size.max(1);

    let be = backend()?;
    let mut loan = be.staging_pairs.lend(&be.ctx)?;
    be.ctx.bind_to_thread()?;
    // SAFETY: `device_ptr_mut` yields the destination's base address and orders
    // the writes on `stream`; `dst.len() >= src_host.len()` (asserted), so every
    // chunk's range lies inside `dst`.
    let (dst_base, _record) = dst.device_ptr_mut(stream);
    // Declared after `loan`, so it drops FIRST: a DMA queued but not yet covered
    // by its buffer's event is drained before the pair can be lent again.
    let mut drain = DrainOnErr {
        stream,
        armed: false,
    };
    let src = src_host.as_ptr() as *const u8;
    let (mut elem_off, mut k) = (0usize, 0usize);
    while elem_off < n_elems {
        let this_elems = (n_elems - elem_off).min(chunk_elems);
        let this_bytes = this_elems * elem_size;
        let byte_off = elem_off * elem_size;
        let buf = loan.buf(k);
        // The DMA that last read this buffer — two chunks back, or the previous
        // borrower's — lands before the host overwrites it.
        buf.sync_event()?;
        // SAFETY: the buffer holds `STAGED_CHUNK_BYTES >= this_bytes`, and its
        // last reader has landed (above), so nothing reads it while it is filled.
        unsafe {
            let copy_t0 = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(src.add(byte_off), buf.ptr as *mut u8, this_bytes);
            STAGING_STATS.note_in(true, this_bytes, copy_t0.elapsed());
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst_base + byte_off as u64,
                buf.ptr as *const core::ffi::c_void,
                this_bytes,
                stream.cu_stream(),
            )
            .result();
            // Armed even on failure: the driver may have queued the copy before
            // reporting the error.
            drain.armed = true;
            r?;
        }
        buf.record_event(stream)?;
        // The buffer's event now covers the DMA, for whoever refills it next.
        drain.armed = false;
        elem_off += this_elems;
        k += 1;
    }
    Ok(())
}

/// [`htod_staged`] for a source page-locked in place: each segment of `plan`
/// ([`crate::registered::direct_plan`]) one `cuMemcpyHtoDAsync` straight from
/// the source, then the stream synchronized, so the call returns with every
/// byte landed, as the staged path's caller may free the source on return.
fn htod_direct<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    src_host: &[T],
    dst: &mut cudarc::driver::CudaViewMut<'_, T>,
    plan: &[crate::registered::Segment],
) -> Result<()> {
    use cudarc::driver::DevicePtrMut;
    let be = backend()?;
    be.ctx.bind_to_thread()?;
    // SAFETY: as in `htod_staged`: `dst.len() >= src_host.len()` (asserted by
    // the caller), and every segment lies inside the source's byte range.
    let (dst_base, _record) = dst.device_ptr_mut(stream);
    let mut drain = DrainOnErr {
        stream,
        armed: false,
    };
    let src = src_host.as_ptr() as *const u8;
    for s in plan {
        // SAFETY: `s.off + s.len <= size_of_val(src_host)` by the plan's
        // construction; the source stays alive until the synchronize below.
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst_base + s.off as u64,
                src.add(s.off) as *const core::ffi::c_void,
                s.len,
                stream.cu_stream(),
            )
            .result()
        };
        drain.armed = true;
        r?;
    }
    stream.synchronize()?;
    drain.armed = false;
    crate::registered::note_direct(std::mem::size_of_val(src_host));
    Ok(())
}

/// Device→host copy of `n` `u64`s from device address `src` into `dst`, through
/// a staging pair of its own: two chunks in flight while the host copies each
/// landed one straight into `dst`'s spare capacity, with no zero-fill pass. `dst`
/// must be empty; on `Ok` it holds exactly the `n` values, and every DMA of this
/// call has landed.
///
/// `after_last_enqueue` runs exactly once, as soon as the LAST chunk's DMA is
/// queued on `stream` — the first moment the caller may queue work that
/// overwrites the source, such as the retained LDE's in-place transpose — and
/// its value is returned. It does not run if an earlier step fails.
///
/// # Safety
/// `src` must address at least `n` `u64`s of one live device allocation whose
/// writers are all ordered before this call on `stream`. Nothing may write them
/// before `after_last_enqueue` runs; work queued on `stream` from then on is
/// ordered behind the reads, and the allocation must outlive them (a
/// stream-ordered free on `stream` does).
pub unsafe fn dtoh_staged_into<R>(
    stream: &Arc<CudaStream>,
    src: cudarc::driver::sys::CUdeviceptr,
    n: usize,
    dst: &mut Vec<u64>,
    after_last_enqueue: impl FnOnce() -> Result<R>,
) -> Result<R> {
    if n == 0 {
        assert!(dst.is_empty(), "dtoh_staged_into: dst must start empty");
        return after_last_enqueue();
    }
    let started = std::time::Instant::now();
    // SAFETY: the caller's contract is `dtoh_staged_uncounted`'s, for `u64`s.
    let result = unsafe { dtoh_staged_uncounted(stream, src, n, dst, after_last_enqueue) }?;
    STAGING_STATS.note_out(true, n * 8, started.elapsed());
    Ok(result)
}

/// [`dtoh_staged_into`] for any plain element type, and counted by its caller
/// instead of in the retained-LDE download's counters: the same pair, chunks and
/// ordering, with `n` counting `T`s. `T`'s size must divide
/// [`STAGED_CHUNK_BYTES`], so no element straddles two chunks.
///
/// # Safety
/// As [`dtoh_staged_into`], with `src` addressing at least `n` `T`s, and every
/// bit pattern the device wrote there a valid `T` (plain data).
pub unsafe fn dtoh_staged_uncounted<T: Copy, R>(
    stream: &Arc<CudaStream>,
    src: cudarc::driver::sys::CUdeviceptr,
    n: usize,
    dst: &mut Vec<T>,
    after_last_enqueue: impl FnOnce() -> Result<R>,
) -> Result<R> {
    assert!(dst.is_empty(), "dtoh_staged_into: dst must start empty");
    let size = std::mem::size_of::<T>();
    assert!(
        size > 0 && STAGED_CHUNK_BYTES.is_multiple_of(size),
        "dtoh_staged_into: the element size must divide the chunk"
    );
    if n == 0 {
        return after_last_enqueue();
    }
    dst.reserve_exact(n);
    let chunk = STAGED_CHUNK_BYTES / size;
    let n_chunks = n.div_ceil(chunk);
    let chunk_len = |k: usize| (n - k * chunk).min(chunk);

    let be = backend()?;
    let mut loan = be.staging_pairs.lend(&be.ctx)?;
    be.ctx.bind_to_thread()?;
    // ⛔ Declared after `loan`, so it drops FIRST, and armed from the first
    // enqueue to the last landing: a DMA WRITES the pinned buffer, and a pair
    // handed back with one in flight would let it land in the next borrower's
    // data. Every early return drains the stream before the pair is lent again.
    let mut drain = DrainOnErr {
        stream,
        armed: false,
    };
    let mut after = Some(after_last_enqueue);
    let mut result = None;
    let mut next = 0usize;
    for k in 0..n_chunks {
        // Keep two chunks in flight: queue up to chunk `k + 1`.
        while next < n_chunks && next < k + 2 {
            let buf = loan.buf(next);
            // Whatever last touched this buffer — this call's chunk `next - 2`,
            // already copied out, or the previous borrower's DMA — is done.
            buf.sync_event()?;
            // SAFETY: the buffer holds `STAGED_CHUNK_BYTES >= chunk_len(next) *
            // size`, nothing reads or writes it (above), and the source range
            // lies inside the caller's `n` values.
            unsafe {
                let r = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    buf.ptr as *mut core::ffi::c_void,
                    src + (next * chunk * size) as u64,
                    chunk_len(next) * size,
                    stream.cu_stream(),
                )
                .result();
                drain.armed = true;
                r?;
            }
            buf.record_event(stream)?;
            next += 1;
            if next == n_chunks {
                let queue_behind = after.take().expect("the last chunk is queued once");
                result = Some(queue_behind()?);
            }
        }
        let buf = loan.buf(k);
        buf.sync_event()?;
        // SAFETY: the event orders this chunk's DMA before the read; `dst` has
        // capacity for `n` values (reserved above) and chunk `k` covers
        // `[k * chunk, k * chunk + chunk_len(k))`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.ptr as *const T,
                dst.as_mut_ptr().add(k * chunk),
                chunk_len(k),
            );
        }
    }
    // Every chunk landed and was copied: nothing of this call is in flight.
    drain.armed = false;
    // SAFETY: the loop wrote all `n` values.
    unsafe { dst.set_len(n) };
    Ok(result.expect("the closure ran at the last enqueue"))
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
    /// The backend whose pool (and context) the event belongs to.
    slot: BackendSlot,
}

impl PooledEvent {
    pub fn event(&self) -> &cudarc::driver::CudaEvent {
        self.event.as_ref().expect("present until drop")
    }
}

impl Drop for PooledEvent {
    fn drop(&mut self) {
        if let (Some(ev), Ok(be)) = (self.event.take(), self.slot.backend()) {
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
        Ok(PooledEvent {
            event: Some(ev),
            slot: BackendSlot::of(self),
        })
    }
}

#[cfg(test)]
mod device_fallback_counter_tests {
    use super::{device_fallbacks, note_device_fallback, reset_device_fallbacks};
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

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

    /// ⛔ THE SEVEN UNDRIVEN SITES' FALSIFIER. The box test drives ONE site
    /// (`DeviceColumns::upload`); this arm is what lets the other seven fail
    /// without seven card fixtures. `note_device_fallback` must be CALLED at
    /// exactly the eight math-cuda argue-surface sites the counter's doc
    /// enumerates — three in `sumcheck.rs`, one in `gkr.rs`, four in
    /// `columns.rs`. Removing the call at ANY site changes the tuple and reddens
    /// this test by name, which localises the loss to the file it happened in.
    /// The ninth site, `multilinear::gpu`'s GKR tree, has its own census there.
    ///
    /// The pattern carries the `crate::device::` prefix so it counts CALLS and
    /// never the definition (`pub fn note_device_fallback`).
    #[test]
    fn note_device_fallback_is_called_at_exactly_the_eight_argue_sites() {
        const PATTERN: &str = "crate::device::note_device_fallback()";
        let sumcheck = include_str!("sumcheck.rs").matches(PATTERN).count();
        let gkr = include_str!("gkr.rs").matches(PATTERN).count();
        let columns = include_str!("columns.rs").matches(PATTERN).count();
        assert_eq!(
            (sumcheck, gkr, columns),
            (3, 1, 4),
            "the argue-surface device-fallback counter must be bumped at exactly \
             the eight sites the counter's doc names: sumcheck ×3, gkr ×1, \
             columns ×4 (found sumcheck {sumcheck}, gkr {gkr}, columns {columns})"
        );
    }

    /// The reservation high-water is a MONOTONE peak: it takes the max, so a
    /// smaller rise never lowers it and a larger one raises it. Card-free — it
    /// exercises `note_reserved` directly (the same call `reserve`/`grow` make on
    /// every successful rise of `be.reserved`), so a peak that failed to track
    /// would show here before any card run relied on the number.
    #[test]
    fn reserved_high_water_keeps_the_max() {
        let peaks = super::LedgerPeaks::default();
        let peak = |p: &super::LedgerPeaks| p.reserved.load(Ordering::Relaxed);
        assert_eq!(peak(&peaks), 0, "a new ledger's peak is zero");
        peaks.note(100);
        assert_eq!(peak(&peaks), 100, "the first rise sets the peak");
        peaks.note(50);
        assert_eq!(peak(&peaks), 100, "a smaller rise must not lower it");
        peaks.note(200);
        assert_eq!(peak(&peaks), 200, "a larger rise raises it");
        assert_eq!(
            peaks.window.load(Ordering::Relaxed),
            200,
            "the window keeps the same peak"
        );
    }
}

#[cfg(test)]
mod staging_report_tests {
    use super::{PinnedStaging, slab_footprint, staging_line};
    use std::sync::Mutex;

    /// A slot claiming `elems` of capacity with no allocation behind it: the
    /// null pointer keeps `Drop` from freeing anything.
    fn slot(elems: usize) -> Mutex<PinnedStaging> {
        let mut s = PinnedStaging::empty();
        s.capacity_elems = elems;
        Mutex::new(s)
    }

    /// Card-free: the footprint sums the allocated slots' bytes and skips empty
    /// ones, counts a slot someone holds as busy instead of waiting for it, and
    /// still reads a poisoned one.
    #[test]
    fn the_slab_footprint_counts_allocated_busy_and_poisoned_slots() {
        let slots = [
            slot(0),
            slot(1 << 20),
            slot(1 << 29),
            slot(1 << 20),
            slot(1 << 22),
        ];
        assert_eq!(slab_footprint(slots[..1].iter()), (0, 0, 0));
        assert_eq!(
            slab_footprint(slots[..3].iter()),
            (((1 << 20) + (1 << 29)) * 8, 2, 0)
        );
        let _held = slots[3].lock().unwrap();
        std::thread::scope(|s| {
            let _ = s
                .spawn(|| {
                    let _g = slots[4].lock().unwrap();
                    panic!("poisoning the slot on purpose");
                })
                .join();
        });
        assert!(slots[4].is_poisoned());
        assert_eq!(
            slab_footprint(slots.iter()),
            (((1 << 20) + (1 << 29) + (1 << 22)) * 8, 3, 1)
        );
    }

    /// The line's shape is a contract with the IDLE-B box script's parsers: four
    /// `<x> GB in` figures (upload shared, staged; download shared, staged), then
    /// the slab segment last, in GiB.
    #[test]
    fn the_staging_line_ends_with_the_shared_slab_footprint() {
        let line = staging_line(6, (4 << 30, 2, 1));
        assert_eq!(line.matches(" GB in ").count(), 4, "{line}");
        assert!(line.contains("| staging pairs 6 of 8 · waits "), "{line}");
        assert!(
            line.ends_with(" | shared slabs pinned 4.00 GiB in 2 slot(s), 1 busy"),
            "{line}"
        );
    }
}

#[cfg(test)]
mod grid_cap_tests {
    use super::{capped_grid, grid_cap, with_grid_cap};

    /// No cap: the full grid. A cap cuts it, never below one block, nests,
    /// and is restored after its closure, a panic included.
    #[test]
    fn a_grid_cap_cuts_the_grid_nests_and_is_restored() {
        assert_eq!((grid_cap(), capped_grid(65_536)), (None, 65_536));
        with_grid_cap(Some(85), || {
            assert_eq!(capped_grid(65_536), 85);
            assert_eq!(capped_grid(12), 12, "a grid under the cap keeps its size");
            with_grid_cap(Some(0), || {
                assert_eq!(capped_grid(65_536), 1, "at least one block")
            });
            with_grid_cap(None, || assert_eq!(capped_grid(65_536), 65_536));
            assert_eq!(grid_cap(), Some(85), "the outer cap restored");
        });
        assert_eq!(grid_cap(), None);
        let caught = std::panic::catch_unwind(|| with_grid_cap(Some(3), || panic!("inside")));
        assert!(caught.is_err());
        assert_eq!(grid_cap(), None, "restored after a panic");
        std::thread::spawn(|| assert_eq!(grid_cap(), None, "a new thread starts uncapped"))
            .join()
            .expect("thread");
    }
}

#[cfg(test)]
mod two_card_openings_tests {
    use super::{
        DeviceRole, MAIN_PLACEMENT, RolePlacement, TwoCardOpenings, current_role, enter_role,
        on_openings_card, placement_for, set_two_card_openings_for_test, two_card_openings,
    };

    const SETTINGS: [TwoCardOpenings; 3] = [
        TwoCardOpenings::Auto,
        TwoCardOpenings::Off,
        TwoCardOpenings::Forced,
    ];

    /// Every role but the openings is card 0's under every setting and card
    /// count, and without a card nothing moves.
    #[test]
    fn the_main_role_and_a_cardless_process_are_card_0_always() {
        for setting in SETTINGS {
            for cards in 0..4 {
                assert_eq!(
                    placement_for(DeviceRole::Main, setting, cards),
                    MAIN_PLACEMENT
                );
            }
            assert_eq!(
                placement_for(DeviceRole::Openings, setting, 0),
                MAIN_PLACEMENT
            );
        }
    }

    /// The openings' table: one card leaves them on card 0's backend unless
    /// forced (a second backend on card 0); two cards or more put them on card
    /// 1 unless off.
    #[test]
    fn the_openings_go_to_card_1_with_two_cards_unless_off() {
        let card1 = RolePlacement {
            slot: 1,
            ordinal: 1,
        };
        let second_on_card0 = RolePlacement {
            slot: 1,
            ordinal: 0,
        };
        let o = DeviceRole::Openings;
        assert_eq!(placement_for(o, TwoCardOpenings::Auto, 1), MAIN_PLACEMENT);
        assert_eq!(placement_for(o, TwoCardOpenings::Off, 1), MAIN_PLACEMENT);
        assert_eq!(
            placement_for(o, TwoCardOpenings::Forced, 1),
            second_on_card0
        );
        for cards in 2..5 {
            assert_eq!(placement_for(o, TwoCardOpenings::Auto, cards), card1);
            assert_eq!(placement_for(o, TwoCardOpenings::Forced, cards), card1);
            assert_eq!(
                placement_for(o, TwoCardOpenings::Off, cards),
                MAIN_PLACEMENT
            );
        }
    }

    #[test]
    fn the_two_card_knob_reads_its_words_and_refuses_the_rest() {
        assert_eq!(
            TwoCardOpenings::parse(None),
            Ok(TwoCardOpenings::Off),
            "off by default"
        );
        assert_eq!(
            TwoCardOpenings::parse(Some("auto")),
            Ok(TwoCardOpenings::Auto)
        );
        assert_eq!(
            TwoCardOpenings::parse(Some(" 0 ")),
            Ok(TwoCardOpenings::Off)
        );
        assert_eq!(
            TwoCardOpenings::parse(Some("1")),
            Ok(TwoCardOpenings::Forced)
        );
        for bad in ["", "on", "2", "yes"] {
            let err = TwoCardOpenings::parse(Some(bad)).expect_err(bad);
            assert!(err.contains("LAMBDA_VM_TWO_CARD_OPENINGS"), "{err}");
        }
    }

    /// A role lasts while its guard lives, nests, and is the thread's alone.
    #[test]
    fn a_role_is_the_threads_and_its_guard_restores_it() {
        assert_eq!(current_role(), DeviceRole::Main);
        {
            let _outer = enter_role(DeviceRole::Openings);
            assert_eq!(current_role(), DeviceRole::Openings);
            std::thread::spawn(|| {
                assert_eq!(current_role(), DeviceRole::Main, "a new thread is Main")
            })
            .join()
            .expect("thread");
            {
                let _inner = enter_role(DeviceRole::Main);
                assert_eq!(current_role(), DeviceRole::Main);
            }
            assert_eq!(current_role(), DeviceRole::Openings);
        }
        assert_eq!(current_role(), DeviceRole::Main);
    }

    /// ★ Card, run alone. Two backends keep their own per-card state: the
    /// BLAKE3 WHIR kernels and the root tables are each backend's own objects;
    /// a reservation made in the openings' role is the second backend's (its
    /// ledger and its peak, not card 0's) and goes back there when dropped on
    /// a card-0 thread. Forced on one card, or on card 1 with two. On one card
    /// the two backends share the device's primary context, so this shows the
    /// caches are per backend, not that card 1's launches work: only a
    /// two-card run shows that.
    #[test]
    #[ignore = "card: run alone"]
    fn each_backend_keeps_its_own_kernels_tables_and_ledger() {
        use super::{BackendSlot, backend, main_backend, reserve};
        let setting = if super::card_count() >= 2 {
            TwoCardOpenings::Auto
        } else {
            TwoCardOpenings::Forced
        };
        set_two_card_openings_for_test(Some(setting));
        let main = main_backend().expect("card 0");
        let second = {
            let _role = enter_role(DeviceRole::Openings);
            backend().expect("the second backend")
        };
        assert!(!std::ptr::eq(main, second), "two backends");
        assert_eq!(BackendSlot::of(second), BackendSlot::Second);
        let k0 = crate::blake3_whir::kernels().expect("card 0's kernels") as *const _;
        let k1 = {
            let _role = enter_role(DeviceRole::Openings);
            crate::blake3_whir::kernels().expect("the second's kernels") as *const _
        };
        assert_ne!(k0, k1, "one set of kernels for two backends");
        let t0 = crate::lde_cm::root_tables(main).expect("card 0's tables") as *const _;
        let t1 = crate::lde_cm::root_tables(second).expect("the second's tables") as *const _;
        assert_ne!(t0, t1, "one set of root tables for two backends");
        const MIB: u64 = 1 << 20;
        let (m0, s0) = (main.reserved_bytes(), second.reserved_bytes());
        let r = std::thread::spawn(|| {
            let _role = enter_role(DeviceRole::Openings);
            super::reset_window_high_water();
            reserve(MIB)
        })
        .join()
        .expect("thread")
        .expect("the second backend has a MiB");
        assert_eq!(
            (main.reserved_bytes(), second.reserved_bytes()),
            (m0, s0 + MIB)
        );
        assert!(
            second
                .peaks
                .window
                .load(std::sync::atomic::Ordering::Relaxed)
                >= s0 + MIB,
            "the second backend's own peak"
        );
        drop(r);
        assert_eq!((main.reserved_bytes(), second.reserved_bytes()), (m0, s0));
        set_two_card_openings_for_test(None);
    }

    /// Off, the openings' work runs as before: on the caller's thread, in its
    /// role, and no rayon pool of their own is made.
    #[test]
    fn off_the_openings_run_on_the_callers_thread_unchanged() {
        set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
        assert!(!two_card_openings());
        let caller = std::thread::current().id();
        let (thread, role) = on_openings_card(|| (std::thread::current().id(), current_role()));
        set_two_card_openings_for_test(None);
        assert_eq!(thread, caller);
        assert_eq!(role, DeviceRole::Main);
    }
}

/// ★ The source scan (the per-card rule): device-bound state
/// lives per backend, never one per process. Host-only: it reads this crate's
/// sources, so a new cache of kernels or device buffers in a plain `static`,
/// a function attribute set outside a backend's start, or a new process-wide
/// `Once` / `OnceLock` fails here before any card run.
#[cfg(test)]
mod device_state_scan_tests {
    /// Every `src/*.rs` of this crate: (file name, text).
    fn sources() -> Vec<(String, String)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut out: Vec<(String, String)> = std::fs::read_dir(&dir)
            .expect("the crate's src")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "rs"))
            .map(|p| {
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (name, std::fs::read_to_string(&p).expect("a source file"))
            })
            .collect();
        out.sort();
        assert!(out.len() >= 30, "found only {} source files", out.len());
        out
    }

    /// Every `static NAME: TYPE` line of `text` outside comments: (name, the
    /// type up to its `=`).
    fn statics(text: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for line in text.lines() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            let rest = ["pub static ", "pub(crate) static ", "static "]
                .iter()
                .find_map(|p| t.strip_prefix(p));
            let Some(rest) = rest else { continue };
            let Some((name, ty)) = rest.split_once(':') else {
                continue;
            };
            let name = name.trim();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            {
                continue;
            }
            let ty = ty.split('=').next().unwrap_or("").trim().to_string();
            out.push((name.to_string(), ty));
        }
        out
    }

    /// Words that name a device-bound object (or a holder of one).
    const DEVICE_TYPES: [&str; 10] = [
        "CudaFunction",
        "CudaModule",
        "CudaSlice",
        "CudaEvent",
        "CUevent",
        "Kernels",
        "Module",
        "RootTables",
        "Staging",
        "Place",
    ];

    fn names_a_word(ty: &str, word: &str) -> bool {
        ty.match_indices(word).any(|(at, _)| {
            let before = ty[..at].chars().next_back();
            let after = ty[at + word.len()..].chars().next();
            !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
                && !after.is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
    }

    /// A process-wide static holding device objects keeps them per backend:
    /// a `PerBackend` cell, an array of one per backend slot, or (listed here,
    /// with why) a pool whose objects record their own context.
    #[test]
    fn no_device_object_in_a_process_wide_static() {
        // The read-back staging pool: each buffer records the context it was
        // made in, and a thread takes only its stream's context's.
        const RECORD_THEIR_OWNER: [(&str, &str); 1] = [("readback.rs", "POOL")];
        let mut breaches = Vec::new();
        for (file, text) in sources() {
            for (name, ty) in statics(&text) {
                if !DEVICE_TYPES.iter().any(|w| names_a_word(&ty, w)) {
                    continue;
                }
                let per_backend = ty.starts_with("crate::device::PerBackend<")
                    || ty.starts_with("PerBackend<")
                    || (ty.starts_with('[') && ty.ends_with("; 2]"));
                if per_backend || RECORD_THEIR_OWNER.contains(&(file.as_str(), name.as_str())) {
                    continue;
                }
                breaches.push(format!("{file}: static {name}: {ty}"));
            }
        }
        assert!(
            breaches.is_empty(),
            "device objects in a process-wide static (make them per backend): {breaches:?}"
        );
    }

    /// A function attribute is set once per card, as its backend starts: only
    /// in `Backend::init_on`.
    #[test]
    fn a_function_attribute_is_set_only_in_init_on() {
        let needle = [".set_", "attribute("].concat();
        for (file, text) in sources() {
            for (at, _) in text.match_indices(&needle) {
                let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
                if text[line_start..at].trim_start().starts_with("//") {
                    continue;
                }
                assert_eq!(
                    file, "device.rs",
                    "an attribute set outside the backend's start"
                );
                let start = text.find("fn init_on(").expect("init_on");
                let end = text[start + 1..]
                    .find("\n    fn ")
                    .into_iter()
                    .chain(text[start + 1..].find("\n    pub fn "))
                    .min()
                    .map_or(text.len(), |e| start + 1 + e);
                assert!(
                    (start..end).contains(&at),
                    "device.rs sets an attribute outside init_on (byte {at})"
                );
            }
        }
    }

    /// Every process-wide `Once`, `OnceLock` and `PerBackend` in the crate, by
    /// file and name: none holds a device object (read when it was added:
    /// knob readings, the card count, the backends themselves, pools of host
    /// threads, host function pointers). A new one fails here until a person
    /// has read it and added it, per the rule.
    const ONCE_CELLS: [(&str, &str); 35] = [
        ("argue_fused.rs", "ON"),
        ("blake3_whir.rs", "KERNELS"),
        ("card_rows.rs", "BYTES"),
        ("card_rows.rs", "COUNTERS"),
        ("card_rows.rs", "FROM_ENV"),
        ("card_rows.rs", "MODULE"),
        ("columns.rs", "ON"),
        ("device.rs", "BACKEND"),
        ("device.rs", "CACHED"),
        ("device.rs", "COUNT"),
        ("device.rs", "EVICTOR"),
        ("device.rs", "OPENINGS_POOL"),
        ("device.rs", "SECOND_BACKEND"),
        ("device.rs", "SETTING"),
        ("device.rs", "WARNED"),
        ("fill.rs", "FROM_ENV"),
        ("fill.rs", "FROM_ENV"),
        ("fill.rs", "MODULE"),
        ("gkr.rs", "ON"),
        ("gkr.rs", "ON"),
        ("grinding.rs", "GRID"),
        ("grinding.rs", "KNOBS"),
        ("lde_cm.rs", "L2"),
        ("lde_cm.rs", "TABLES"),
        ("nvtx.rs", "API"),
        ("nvtx.rs", "API"),
        ("p1_stark.rs", "KERNELS"),
        ("p1w16.rs", "MODULE"),
        ("pinned.rs", "ARENAS_MADE"),
        ("pinned.rs", "ONCE"),
        ("pinned.rs", "POOL_GIB"),
        ("stream_class.rs", "SETTING"),
        ("whir.rs", "ENABLED"),
        ("whir.rs", "INSTALLED"),
        ("whir_coset.rs", "MODULE"),
    ];

    #[test]
    fn every_once_cell_is_on_the_pinned_list() {
        let mut found: Vec<(String, String)> = Vec::new();
        for (file, text) in sources() {
            for (name, ty) in statics(&text) {
                if names_a_word(&ty, "Once")
                    || ty.contains("OnceLock<")
                    || ty.contains("PerBackend<")
                {
                    found.push((file.clone(), name));
                }
            }
        }
        found.sort();
        let mut pinned: Vec<(String, String)> = ONCE_CELLS
            .iter()
            .map(|&(f, n)| (f.to_string(), n.to_string()))
            .collect();
        pinned.sort();
        assert_eq!(
            found, pinned,
            "the crate's process-wide Once / OnceLock / PerBackend cells changed: read the new \
             one against the per-card rule, then pin it"
        );
    }
}
