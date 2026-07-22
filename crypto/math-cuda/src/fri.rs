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

/// A FRI layer's folded evaluations kept resident on device. `buf` is the
/// layer's ext3 evals **interleaved** (`3 * len` u64, `[a0,b0,c0, a1,b1,c1, …]`),
/// and `len` is the number of ext3 evals in the layer (`n_out`), carried
/// explicitly so the query phase has the metadata even once the host `Vec` is
/// dropped (Step F2). Freed when the `buf` Arc drops.
#[derive(Clone)]
pub struct GpuFriEvals {
    pub buf: Arc<CudaSlice<u64>>,
    pub len: usize,
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
        assert!(n0 >= 2 && n0.is_power_of_two());
        assert_eq!(evals_host.len(), 3 * n0);
        assert_eq!(inv_tw_host.len(), n0 / 2);

        let be = backend()?;
        let stream = be.next_stream();

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
        retain_evals: bool,
    ) -> Result<(Vec<u64>, crate::lde::GpuMerkleTree, Option<GpuFriEvals>)> {
        #[cfg(feature = "test-faults")]
        check_fault_injection()?;
        let be = backend()?;
        let n_in = self.current_n;
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
        let zeta_dev = self.stream.clone_htod(&zeta_raw)?;

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
        build_inner_tree_levels(self.stream.as_ref(), be, &mut nodes_dev, num_leaves)?;

        // Update inv_twiddles for the next layer: `new[j] = old[2j]^2` for
        // j in 0..n_out/2. (If n_out == 1, skip; no next fold.) Writes into
        // a fresh device buffer to avoid the cross-thread race the in-place
        // version had (thread j reads old[2j] while thread 2j writes old[2j]).
        let tw_next = n_out / 2;
        if tw_next > 0 {
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
        self.stream.synchronize()?;

        // The folded output buffer for this layer (3 * n_out ext3 u64).
        let out_view = if self.a_is_input {
            self.evals_b.slice(0..3 * n_out)
        } else {
            self.evals_a.slice(0..3 * n_out)
        };
        // Host copy — still authoritative until Step F2 routes openings to device.
        let layer_evals: Vec<u64> = self.stream.clone_dtoh(&out_view)?;
        crate::stagebytes::add_fri_layer_d2h(layer_evals.len() * 8);
        // F0: for committed (queried) layers, retain the folded evals on device in
        // a fresh buffer (the ping-pong scratch `evals_a`/`evals_b` is overwritten
        // by later folds) so the query phase can gather opened values on device.
        // The terminal fold is never queried, so `retain_evals` is false there and
        // this D2D copy is skipped.
        let retained_evals = if retain_evals {
            Some(GpuFriEvals {
                buf: Arc::new(self.stream.clone_dtod(&out_view)?),
                len: n_out,
            })
        } else {
            None
        };

        // Keep the layer tree resident on device; copy only the 32-byte root so
        // R4 query openings gather paths on device instead of copying the tree.
        let mut root = [0u8; 32];
        self.stream
            .memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
        self.stream.synchronize()?;

        self.a_is_input = !self.a_is_input;
        self.current_n = n_out;

        let tree = crate::lde::GpuMerkleTree {
            nodes: std::sync::Arc::new(nodes_dev),
            leaves_len: num_leaves,
            root,
        };
        Ok((layer_evals, tree, retained_evals))
    }

    // (retained_evals is `Some` only when `retain_evals` is passed for a
    // committed layer; the terminal fold passes false and gets `None`.)
}
