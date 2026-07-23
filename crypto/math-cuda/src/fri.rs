//! Fully-device-resident FRI commit phase orchestration.
//!
//! The host loop (in the stark crate) samples each layer's `zeta` from the
//! transcript and feeds it in. This module keeps the folded evaluations,
//! twiddles, and per-layer Merkle trees on device, only D2H'ing each
//! layer's root (to append to the transcript), plus its full evals and
//! tree nodes (to plug into `FriLayer` for the query phase).
//!
//! Mirrors `commit_phase_from_evaluations` at
//! `crypto/stark/src/fri/mod.rs`.

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::Result;
use crate::device::backend;
use crate::merkle::build_inner_tree_levels;

/// Test-only fault injection. When the `test-faults` feature is on, setting
/// this to a finite value forces the next `fold_and_commit_layer` call to
/// return Err and decrement the counter. Tests use this to exercise the
/// CPU-fallback path in `try_fri_commit_gpu`.
#[cfg(feature = "test-faults")]
pub static FAULT_FOLDS_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_FOLDS_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let new = FAULT_FOLDS_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if new == 0 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

/// Device-side state across FRI commit iterations. Owns two ext3 eval
/// buffers (flip-flopped as layer input / output) and the inv_twiddles
/// buffer. Freed when dropped.
pub struct FriCommitState {
    pub stream: Arc<CudaStream>,
    // Ping-pong evaluation buffers. Both sized `3 * n0` u64 at init. Each
    // successive fold uses half the space. Cheap to pre-allocate vs. per-
    // layer alloc.
    evals_a: CudaSlice<u64>,
    evals_b: CudaSlice<u64>,
    /// Base-field inv_twiddles; `n0 / 2` u64 at init, halved each layer.
    inv_tw: CudaSlice<u64>,
    /// Number of ext3 elements in the buffer currently acting as fold input
    /// (`evals_a` or `evals_b`, selected by `a_is_input`).
    pub current_n: usize,
    /// Which buffer holds the current layer's input. Toggles each fold.
    a_is_input: bool,
}

impl FriCommitState {
    /// H2D the starting evals (ext3 interleaved, 3 * n0 u64) and the
    /// initial inv_twiddles (base field, n0/2 u64). `n0` must be a power of
    /// two and >= 2.
    pub fn new(evals_host: &[u64], inv_tw_host: &[u64], n0: usize) -> Result<Self> {
        crate::nvtx_range!("gpu:fri_commit_new");
        assert!(n0 >= 2 && n0.is_power_of_two());
        assert_eq!(evals_host.len(), 3 * n0);
        assert_eq!(inv_tw_host.len(), n0 / 2);

        let be = backend()?;
        let stream = be.next_stream();
        crate::gpu_span!(&stream, "gpu:fri_commit_new");

        // SAFETY: every byte of evals_a is overwritten by the H2D below.
        // evals_b is written by the first fold before it is read.
        let mut evals_a = unsafe { stream.alloc::<u64>(3 * n0) }?;
        let evals_b = unsafe { stream.alloc::<u64>(3 * n0) }?;
        stream.memcpy_htod(evals_host, &mut evals_a)?;
        let inv_tw = stream.clone_htod(inv_tw_host)?;

        Ok(Self {
            stream,
            evals_a,
            evals_b,
            inv_tw,
            current_n: n0,
            a_is_input: true,
        })
    }

    /// Fold the current layer using `zeta`, run the row-pair Keccak leaves
    /// + pair-hash Merkle tree kernels on the result, and D2H:
    ///   - the new root (32 bytes)
    ///   - the new layer's evals (3 * (current_n / 2) u64s)
    ///   - the new layer's Merkle tree nodes (standard layout, byte-packed)
    ///
    /// Also advances the internal twiddle factors for the next layer.
    pub fn fold_and_commit_layer(
        &mut self,
        zeta_raw: [u64; 3],
    ) -> Result<(Vec<u64>, crate::lde::GpuMerkleTree)> {
        crate::nvtx_range!("gpu:fri_fold_and_commit_layer");
        crate::gpu_span!(&self.stream, "gpu:fri_layer");
        #[cfg(feature = "test-faults")]
        check_fault_injection()?;
        let be = backend()?;
        let n_in = self.current_n;
        crate::nvtx_range!("fri_layer:{}", n_in);
        let n_out = n_in / 2;
        // n_out == 1 (terminal_len < 2) never reaches this path: `try_fri_commit_gpu`
        // filters it out and returns None so the CPU fallback handles it.
        assert!(
            n_out >= 2,
            "fold_and_commit_layer requires n_out >= 2 (n_out == 1 falls back to the CPU path)"
        );

        // Row-pair leaves: each leaf hashes two consecutive ext3 evals.
        let num_leaves = n_out / 2;
        let tight_total_nodes = 2 * num_leaves - 1;

        // H2D zeta.
        let zeta_dev = {
            crate::nvtx_range!("h2d");
            self.stream.clone_htod(&zeta_raw)?
        };

        let cfg = LaunchConfig {
            grid_dim: ((n_out as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_out_u64 = n_out as u64;

        // Split the eval buffers into (input, output) based on a_is_input.
        // Disjoint-field borrow is fine since evals_a and evals_b are
        // separate fields.
        let (input_evals, output_evals): (&CudaSlice<u64>, &mut CudaSlice<u64>) = if self.a_is_input
        {
            (&self.evals_a, &mut self.evals_b)
        } else {
            (&self.evals_b, &mut self.evals_a)
        };
        unsafe {
            crate::nvtx_range!("fold");
            self.stream
                .launch_builder(&be.fri_fold_ext3)
                .arg(input_evals)
                .arg(&n_out_u64)
                .arg(&self.inv_tw)
                .arg(&zeta_dev)
                .arg(output_evals)
                .launch(cfg)?;
        }

        // SAFETY: keccak_fri_leaves_ext3 writes the leaves [num_leaves-1, 2*num_leaves-1)
        // and build_inner_tree_levels writes every inner node [0, num_leaves-1), so all
        // tight_total_nodes * 32 bytes are initialised before the D2H below reads them.
        let mut nodes_dev = unsafe { self.stream.alloc::<u8>(tight_total_nodes * 32) }?;
        let leaves_offset_bytes = (num_leaves - 1) * 32;
        {
            crate::nvtx_range!("keccak_leaves");
            let mut leaves_view =
                nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
            let num_leaves_u64 = num_leaves as u64;
            let grid = (num_leaves as u32).div_ceil(128);
            let kcfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // Leaves read from the layer's OUTPUT eval buffer (the buffer
            // we just wrote to above).
            let output_evals: &CudaSlice<u64> = if self.a_is_input {
                &self.evals_b
            } else {
                &self.evals_a
            };
            unsafe {
                self.stream
                    .launch_builder(&be.keccak_fri_leaves_ext3)
                    .arg(output_evals)
                    .arg(&num_leaves_u64)
                    .arg(&mut leaves_view)
                    .launch(kcfg)?;
            }
        }
        {
            crate::nvtx_range!("tree_levels");
            build_inner_tree_levels(self.stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
        }

        // Update inv_twiddles for the next layer: `new[j] = old[2j]^2` for
        // j in 0..n_out/2. (If n_out == 1, skip; no next fold.) Writes into
        // a fresh device buffer to avoid the cross-thread race the in-place
        // version had (thread j reads old[2j] while thread 2j writes old[2j]).
        let tw_next = n_out / 2;
        if tw_next > 0 {
            crate::nvtx_range!("tw_update");
            let mut tw_out = unsafe { self.stream.alloc::<u64>(tw_next) }?;
            let grid = (tw_next as u32).div_ceil(128);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let tw_next_u64 = tw_next as u64;
            unsafe {
                self.stream
                    .launch_builder(&be.fri_update_twiddles)
                    .arg(&self.inv_tw)
                    .arg(&mut tw_out)
                    .arg(&tw_next_u64)
                    .launch(cfg)?;
            }
            self.inv_tw = tw_out;
        }

        // Sync and D2H.
        {
            crate::nvtx_range!("sync");
            crate::timing::timed_sync(&self.stream, "gpu:fri_layer")?;
        }

        // Layer evals: 3 * n_out u64 from the output buffer, staged through
        // the per-worker pinned slab (async DMA) instead of a blocking
        // pageable copy. The wait is deferred past the root copy below.
        let n_evals = 3 * n_out;
        let pending = {
            crate::nvtx_range!("d2h");
            let output_evals: &CudaSlice<u64> = if self.a_is_input {
                &self.evals_b
            } else {
                &self.evals_a
            };
            crate::device::async_dtoh_via(
                &self.stream,
                be.pinned_staging(),
                &be.ctx,
                output_evals,
                n_evals,
            )?
        };

        // Keep the layer tree resident on device; copy only the 32-byte root so
        // R4 query openings gather paths on device instead of copying the tree.
        // This pageable copy drains the stream (including the evals DMA above),
        // so the pending wait after it is instant — one block covers both.
        let mut root = [0u8; 32];
        {
            crate::nvtx_range!("d2h");
            self.stream
                .memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
        }
        let mut layer_evals = vec![0u64; n_evals];
        pending.wait_into_u64(&mut layer_evals)?;

        self.a_is_input = !self.a_is_input;
        self.current_n = n_out;

        let tree = crate::lde::GpuMerkleTree {
            nodes: std::sync::Arc::new(nodes_dev),
            leaves_len: num_leaves,
            root,
        };
        Ok((layer_evals, tree))
    }
}
