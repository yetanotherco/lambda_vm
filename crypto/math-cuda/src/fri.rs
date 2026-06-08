//! Fully-device-resident FRI commit phase orchestration.
//!
//! The host loop (in the stark crate) samples each layer's `zeta` from the
//! transcript and feeds it in; this module keeps the folded evaluations,
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

/// Device-side state across FRI commit iterations. Owns two ext3 eval
/// buffers (flip-flopped as layer input / output) and the inv_twiddles
/// buffer. Freed when dropped.
pub struct FriCommitState {
    pub stream: Arc<CudaStream>,
    // Ping-pong evaluation buffers. Both sized `3 * n0` u64 at init; each
    // successive fold uses half the space. Cheap to pre-allocate vs. per-
    // layer alloc.
    evals_a: CudaSlice<u64>,
    evals_b: CudaSlice<u64>,
    /// Base-field inv_twiddles; `n0 / 2` u64 at init, halved each layer.
    inv_tw: CudaSlice<u64>,
    /// Number of ext3 elements currently in the "input" buffer.
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
    /// Also updates `inv_twiddles` in place to shrink for the next layer.
    pub fn fold_and_commit_layer(
        &mut self,
        zeta_raw: [u64; 3],
    ) -> Result<(Vec<u8>, Vec<u64>, Vec<u8>)> {
        let be = backend()?;
        let n_in = self.current_n;
        let n_out = n_in / 2;
        // fold_final handles the n_out == 1 last layer (no Merkle commit).
        assert!(
            n_out >= 2,
            "fold_and_commit_layer requires n_out >= 2; use fold_final"
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

        if self.a_is_input {
            unsafe {
                self.stream
                    .launch_builder(&be.fri_fold_ext3)
                    .arg(&self.evals_a)
                    .arg(&n_out_u64)
                    .arg(&self.inv_tw)
                    .arg(&zeta_dev)
                    .arg(&mut self.evals_b)
                    .launch(cfg)?;
            }
        } else {
            unsafe {
                self.stream
                    .launch_builder(&be.fri_fold_ext3)
                    .arg(&self.evals_b)
                    .arg(&n_out_u64)
                    .arg(&self.inv_tw)
                    .arg(&zeta_dev)
                    .arg(&mut self.evals_a)
                    .launch(cfg)?;
            }
        }

        // Keccak leaves + pair-hash tree into fresh device buffer.
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
            // Leaves read from the layer's OUTPUT eval buffer.
            if self.a_is_input {
                unsafe {
                    self.stream
                        .launch_builder(&be.keccak_fri_leaves_ext3)
                        .arg(&self.evals_b)
                        .arg(&num_leaves_u64)
                        .arg(&mut leaves_view)
                        .launch(kcfg)?;
                }
            } else {
                unsafe {
                    self.stream
                        .launch_builder(&be.keccak_fri_leaves_ext3)
                        .arg(&self.evals_a)
                        .arg(&num_leaves_u64)
                        .arg(&mut leaves_view)
                        .launch(kcfg)?;
                }
            }
        }
        {
            let mut level_begin: u64 = (num_leaves - 1) as u64;
            while level_begin != 0 {
                let new_begin = level_begin / 2;
                let n_pairs = level_begin - new_begin;
                let grid = (n_pairs as u32).div_ceil(128);
                let cfg = LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.stream
                        .launch_builder(&be.keccak_merkle_level)
                        .arg(&mut nodes_dev)
                        .arg(&new_begin)
                        .arg(&n_pairs)
                        .launch(cfg)?;
                }
                level_begin = new_begin;
            }
        }

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

        // Layer evals: 3 * n_out u64 from the output buffer.
        let layer_evals: Vec<u64> = if self.a_is_input {
            let view = self.evals_b.slice(0..3 * n_out);
            self.stream.clone_dtoh(&view)?
        } else {
            let view = self.evals_a.slice(0..3 * n_out);
            self.stream.clone_dtoh(&view)?
        };

        // Tree nodes.
        let nodes_bytes: Vec<u8> = self.stream.clone_dtoh(&nodes_dev)?;
        debug_assert_eq!(nodes_bytes.len(), tight_total_nodes * 32);

        let mut root = vec![0u8; 32];
        root.copy_from_slice(&nodes_bytes[0..32]);

        self.a_is_input = !self.a_is_input;
        self.current_n = n_out;

        Ok((root, layer_evals, nodes_bytes))
    }

    /// Final fold, no Merkle commit. Returns the single ext3 output
    /// element (the FRI last_value).
    pub fn fold_final(&mut self, zeta_raw: [u64; 3]) -> Result<[u64; 3]> {
        let be = backend()?;
        let n_in = self.current_n;
        let n_out = n_in / 2;
        assert!(n_out >= 1);

        let zeta_dev = self.stream.clone_htod(&zeta_raw)?;
        let cfg = LaunchConfig {
            grid_dim: ((n_out as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_out_u64 = n_out as u64;

        if self.a_is_input {
            unsafe {
                self.stream
                    .launch_builder(&be.fri_fold_ext3)
                    .arg(&self.evals_a)
                    .arg(&n_out_u64)
                    .arg(&self.inv_tw)
                    .arg(&zeta_dev)
                    .arg(&mut self.evals_b)
                    .launch(cfg)?;
            }
        } else {
            unsafe {
                self.stream
                    .launch_builder(&be.fri_fold_ext3)
                    .arg(&self.evals_b)
                    .arg(&n_out_u64)
                    .arg(&self.inv_tw)
                    .arg(&zeta_dev)
                    .arg(&mut self.evals_a)
                    .launch(cfg)?;
            }
        }

        self.stream.synchronize()?;
        let out_first: Vec<u64> = if self.a_is_input {
            let view = self.evals_b.slice(0..3);
            self.stream.clone_dtoh(&view)?
        } else {
            let view = self.evals_a.slice(0..3);
            self.stream.clone_dtoh(&view)?
        };
        self.a_is_input = !self.a_is_input;
        self.current_n = n_out;
        Ok([out_first[0], out_first[1], out_first[2]])
    }
}
