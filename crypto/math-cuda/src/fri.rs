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

use crate::DeviceHash;
use crate::Result;
use crate::device::backend;

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

/// Device-side state across FRI commit iterations. Owns the current fold
/// input (the previous layer's evals) and the inv_twiddles buffer. The input
/// is an `Arc` because the caller may also retain it as that layer's
/// `gpu_evals` — it does so only on the device-only path, where no host copy
/// of the evals exists. Freed when the last holder drops.
pub struct FriCommitState {
    pub stream: Arc<CudaStream>,
    /// Which hash family commits each layer's tree.
    hash: DeviceHash,
    /// Current fold input. Each fold allocates a fresh output buffer that is
    /// both returned to the caller (kept resident for the query phase) and
    /// becomes the next fold's input.
    current: Arc<CudaSlice<u64>>,
    /// Base-field inv_twiddles; `n0 / 2` u64 at init, halved each layer.
    inv_tw: CudaSlice<u64>,
    /// Number of ext3 elements in `current`.
    pub current_n: usize,
}

impl FriCommitState {
    /// H2D the starting evals (ext3 interleaved, 3 * n0 u64) and the
    /// initial inv_twiddles (base field, n0/2 u64). `n0` must be a power of
    /// two and >= 2.
    pub fn new(
        evals_host: &[u64],
        inv_tw_host: &[u64],
        n0: usize,
        hash: DeviceHash,
    ) -> Result<Self> {
        assert!(n0 >= 2 && n0.is_power_of_two());
        assert_eq!(evals_host.len(), 3 * n0);
        assert_eq!(inv_tw_host.len(), n0 / 2);

        let be = backend()?;
        let stream = be.next_stream();

        // SAFETY: every byte of evals is overwritten by the H2D below.
        let mut evals = unsafe { stream.alloc::<u64>(3 * n0) }?;
        stream.memcpy_htod(evals_host, &mut evals)?;
        let inv_tw = stream.clone_htod(inv_tw_host)?;

        Ok(Self {
            stream,
            hash,
            current: Arc::new(evals),
            inv_tw,
            current_n: n0,
        })
    }

    /// Like [`Self::new`], but adopts a device-resident codeword (already in
    /// FRI bit-reversed order) and its producing stream — no evals H2D.
    pub fn new_dev(
        codeword: crate::deep::GpuDeepCodeword,
        inv_tw_host: &[u64],
        hash: DeviceHash,
    ) -> Result<Self> {
        let crate::deep::GpuDeepCodeword { buf, n, stream } = codeword;
        assert!(n >= 2 && n.is_power_of_two());
        assert_eq!(buf.len(), 3 * n);
        assert_eq!(inv_tw_host.len(), n / 2);

        let inv_tw = stream.clone_htod(inv_tw_host)?;

        Ok(Self {
            stream,
            hash,
            current: Arc::new(buf),
            inv_tw,
            current_n: n,
        })
    }

    /// Fold the current layer using `zeta`, run the row-pair leaf kernels of
    /// the configured hash family and
    /// pair-hash Merkle tree kernels on the result, and return the layer's
    /// evals — device-resident Arc, plus a host copy only when `want_host` —
    /// with its resident Merkle tree (root D2H'd, 32 bytes).
    ///
    /// Also advances the internal twiddle factors for the next layer.
    #[allow(clippy::type_complexity)]
    pub fn fold_and_commit_layer(
        &mut self,
        zeta_raw: [u64; 3],
        want_host: bool,
    ) -> Result<(
        Option<Vec<u64>>,
        Arc<CudaSlice<u64>>,
        crate::lde::GpuMerkleTree,
    )> {
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

        // Fresh output buffer per layer: it is retained by the caller for the
        // query phase and becomes the next fold's input.
        // SAFETY: the fold kernel writes all 3 * n_out slots before any read.
        let mut out = unsafe { self.stream.alloc::<u64>(3 * n_out) }?;
        let input_evals: &CudaSlice<u64> = self.current.as_ref();
        unsafe {
            self.stream
                .launch_builder(&be.fri_fold_ext3)
                .arg(input_evals)
                .arg(&n_out_u64)
                .arg(&self.inv_tw)
                .arg(&zeta_dev)
                .arg(&mut out)
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
            match self.hash {
                DeviceHash::Keccak256 => {
                    let grid = (num_leaves as u32).div_ceil(128);
                    let kcfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.stream
                            .launch_builder(&be.keccak_fri_leaves_ext3)
                            .arg(&out)
                            .arg(&num_leaves_u64)
                            .arg(&mut leaves_view)
                            .launch(kcfg)?;
                    }
                }
                DeviceHash::Blake3 => crate::blake3::launch_fri_leaves_ext3(
                    self.stream.as_ref(),
                    be,
                    &out,
                    num_leaves_u64,
                    &mut leaves_view,
                )?,
                DeviceHash::Rpx256 => crate::rpx::launch_fri_leaves_ext3(
                    self.stream.as_ref(),
                    be,
                    &out,
                    num_leaves_u64,
                    &mut leaves_view,
                )?,
                DeviceHash::Rpo256 | DeviceHash::Poseidon => unimplemented!(
                    "{:?} device commit not yet ported (FRI layer ext3 leaves)",
                    self.hash
                ),
            }
        }
        match self.hash {
            DeviceHash::Keccak256 => crate::merkle::build_inner_tree_levels(
                self.stream.as_ref(),
                be,
                &mut nodes_dev,
                num_leaves,
                DeviceHash::Keccak256,
            )?,
            DeviceHash::Blake3 => crate::blake3::build_inner_tree_levels(
                self.stream.as_ref(),
                be,
                &mut nodes_dev,
                num_leaves,
            )?,
            DeviceHash::Rpx256 => crate::rpx::build_inner_tree_levels(
                self.stream.as_ref(),
                be,
                &mut nodes_dev,
                num_leaves,
            )?,
            DeviceHash::Rpo256 | DeviceHash::Poseidon => unimplemented!(
                "{:?} device commit not yet ported (FRI layer inner tree levels)",
                self.hash
            ),
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

        // Layer evals to host only when a host copy is wanted (fallback
        // consumers), staged through the per-worker pinned slab (async DMA);
        // the wait is deferred past the root copy below.
        let n_evals = 3 * n_out;
        let pending = if want_host {
            Some(crate::device::async_dtoh_via(
                &self.stream,
                be.pinned_staging(),
                &be.ctx,
                &out,
                n_evals,
            )?)
        } else {
            None
        };

        // Keep the layer tree resident on device; copy only the 32-byte root so
        // R4 query openings gather paths on device instead of copying the tree.
        // This pageable copy drains the stream (including any evals DMA above),
        // so the pending wait after it is instant — one block covers both.
        let mut root = [0u8; 32];
        self.stream
            .memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
        let layer_evals = match pending {
            Some(p) => {
                let mut v = vec![0u64; n_evals];
                p.wait_into_u64(&mut v)?;
                Some(v)
            }
            None => None,
        };

        let out = Arc::new(out);
        self.current = Arc::clone(&out);
        self.current_n = n_out;

        let tree = crate::lde::GpuMerkleTree {
            nodes: std::sync::Arc::new(nodes_dev),
            leaves_len: num_leaves,
            root,
        };
        Ok((layer_evals, out, tree))
    }

    /// One binary fold of the current codeword with `zeta_raw` into a fresh
    /// buffer (the same `fri_fold_ext3` launch as [`Self::fold_and_commit_layer`]),
    /// then the twiddle update for the halved domain. The output becomes the
    /// current codeword; the input is released once no caller holds it.
    fn fold_once(&mut self, be: &crate::device::Backend, zeta_raw: [u64; 3]) -> Result<()> {
        let n_out = self.current_n / 2;
        assert!(n_out >= 1, "fold_once: nothing left to fold");
        let zeta_dev = self.stream.clone_htod(&zeta_raw)?;
        let cfg = LaunchConfig {
            grid_dim: ((n_out as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_out_u64 = n_out as u64;
        // SAFETY: the fold kernel writes all 3 * n_out slots before any read.
        let mut out = unsafe { self.stream.alloc::<u64>(3 * n_out) }?;
        unsafe {
            self.stream
                .launch_builder(&be.fri_fold_ext3)
                .arg(self.current.as_ref())
                .arg(&n_out_u64)
                .arg(&self.inv_tw)
                .arg(&zeta_dev)
                .arg(&mut out)
                .launch(cfg)?;
        }
        // `new[j] = old[2j]^2` into a fresh buffer (see `fold_and_commit_layer`
        // for why not in place).
        let tw_next = n_out / 2;
        if tw_next > 0 {
            // SAFETY: the update kernel writes all tw_next slots.
            let mut tw_out = unsafe { self.stream.alloc::<u64>(tw_next) }?;
            let cfg = LaunchConfig {
                grid_dim: ((tw_next as u32).div_ceil(128), 1, 1),
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
        self.current = Arc::new(out);
        self.current_n = n_out;
        Ok(())
    }

    /// The S3 (higher-arity committed FRI) step: fold the current codeword
    /// `zeta_powers.len()` times — fold `ℓ` with `zeta_powers[ℓ]`, which the
    /// caller sets to `ζ^{2^ℓ}` — then commit the result as a layer whose leaf
    /// `g` hashes the `2^group_log` consecutive ext3 values
    /// `[g·2^group_log, (g+1)·2^group_log)` (the configured hash's `Batched`
    /// leaf over the group), with the pair-hash inner tree on top.
    ///
    /// The fold count and the group size are separate on purpose: committed
    /// layer `j` is reached by the PREVIOUS layer's `d_{j−1}` folds and grouped
    /// by its own `d_j` (FRI.md §3.1). Only the last fold's output is kept; the
    /// intermediate codewords are released as the chain advances.
    ///
    /// Returns what [`Self::fold_and_commit_layer`] returns: the layer's evals
    /// (host copy only when `want_host`), the resident evals, and the resident
    /// tree with its root D2H'd.
    #[allow(clippy::type_complexity)]
    pub fn fold_and_commit_group(
        &mut self,
        zeta_powers: &[[u64; 3]],
        group_log: u32,
        want_host: bool,
    ) -> Result<(
        Option<Vec<u64>>,
        Arc<CudaSlice<u64>>,
        crate::lde::GpuMerkleTree,
    )> {
        #[cfg(feature = "test-faults")]
        check_fault_injection()?;
        let be = backend()?;
        for &z in zeta_powers {
            self.fold_once(be, z)?;
        }
        let n = self.current_n;
        assert!(
            group_log >= 1 && (n >> group_log) >= 2 && (n >> group_log) << group_log == n,
            "fold_and_commit_group: a layer of {n} values cannot hold >= 2 groups of 2^{group_log}"
        );
        let num_leaves = n >> group_log;
        let nodes_dev = commit_group_leaves(
            &self.stream,
            be,
            self.hash,
            self.current.as_ref(),
            num_leaves,
            1u64 << group_log,
        )?;

        let n_evals = 3 * n;
        let pending = if want_host {
            Some(crate::device::async_dtoh_via(
                &self.stream,
                be.pinned_staging(),
                &be.ctx,
                self.current.as_ref(),
                n_evals,
            )?)
        } else {
            None
        };
        // The pageable root copy drains the stream, the evals DMA included.
        let mut root = [0u8; 32];
        self.stream
            .memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
        let layer_evals = match pending {
            Some(p) => {
                let mut v = vec![0u64; n_evals];
                p.wait_into_u64(&mut v)?;
                Some(v)
            }
            None => None,
        };
        let tree = crate::lde::GpuMerkleTree {
            nodes: Arc::new(nodes_dev),
            leaves_len: num_leaves,
            root,
        };
        Ok((layer_evals, Arc::clone(&self.current), tree))
    }

    /// Fold the current codeword `zeta_powers.len()` times (fold `ℓ` with
    /// `zeta_powers[ℓ]`) and copy the result to the host, with no commitment:
    /// the S3 fold into the terminal codeword after the last committed layer.
    pub fn fold_to_host(&mut self, zeta_powers: &[[u64; 3]]) -> Result<Vec<u64>> {
        #[cfg(feature = "test-faults")]
        check_fault_injection()?;
        let be = backend()?;
        for &z in zeta_powers {
            self.fold_once(be, z)?;
        }
        let out = self.stream.clone_dtoh(self.current.as_ref())?;
        self.stream.synchronize()?;
        Ok(out)
    }
}

/// Hash `num_leaves` group leaves of `group` consecutive ext3 values each from
/// the interleaved `evals` (`3 · num_leaves · group` u64) and build the inner
/// tree on top: the full `(2·num_leaves − 1) · 32`-byte node buffer, root at 0.
fn commit_group_leaves(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    hash: DeviceHash,
    evals: &CudaSlice<u64>,
    num_leaves: usize,
    group: u64,
) -> Result<CudaSlice<u8>> {
    assert!(num_leaves >= 2 && num_leaves.is_power_of_two());
    assert!(evals.len() as u64 >= 3 * num_leaves as u64 * group);
    let tight_total_nodes = 2 * num_leaves - 1;
    // SAFETY: the leaf kernel writes the leaves [num_leaves-1, 2*num_leaves-1)
    // and the inner-level walk every node [0, num_leaves-1) before any read.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(tight_total_nodes * 32) }?;
    let leaves_offset_bytes = (num_leaves - 1) * 32;
    {
        let mut leaves_view =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let (kernel, cfg) = match hash {
            DeviceHash::Keccak256 => (
                &be.keccak_fri_group_leaves_ext3,
                crate::merkle::keccak_launch_cfg(num_leaves_u64),
            ),
            DeviceHash::Blake3 => (
                &be.blake3_fri_group_leaves_ext3,
                crate::blake3::blake3_launch_cfg(num_leaves_u64),
            ),
            DeviceHash::Rpx256 => (
                &be.rpx_fri_group_leaves_ext3,
                crate::rpx::rpx_launch_cfg(num_leaves_u64),
            ),
            DeviceHash::Rpo256 | DeviceHash::Poseidon => {
                unimplemented!("{hash:?} device commit not yet ported (FRI group leaves)")
            }
        };
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(evals)
                .arg(&num_leaves_u64)
                .arg(&group)
                .arg(&mut leaves_view)
                .launch(cfg)?;
        }
    }
    match hash {
        DeviceHash::Keccak256 => crate::merkle::build_inner_tree_levels(
            stream.as_ref(),
            be,
            &mut nodes_dev,
            num_leaves,
            DeviceHash::Keccak256,
        )?,
        DeviceHash::Blake3 => {
            crate::blake3::build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?
        }
        DeviceHash::Rpx256 => {
            crate::rpx::build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?
        }
        DeviceHash::Rpo256 | DeviceHash::Poseidon => {
            unimplemented!("{hash:?} device commit not yet ported (FRI group inner tree levels)")
        }
    }
    Ok(nodes_dev)
}

/// Parity harness (not a production path): commit an interleaved ext3 eval
/// vector as an S3 group-leaf FRI layer — leaf `g` = the `2^group_log`
/// consecutive values from `g·2^group_log` — under `hash`, and return the full
/// host node buffer (`(2·num_leaves − 1) · 32` bytes, standard layout) so tests
/// can compare it node for node with the host tree. Production commits through
/// [`FriCommitState::fold_and_commit_group`], over the same kernels.
pub fn build_fri_group_tree_from_evals_ext3(
    evals: &[u64],
    group_log: u32,
    hash: DeviceHash,
) -> Result<Vec<u8>> {
    assert!(evals.len().is_multiple_of(3));
    let n = evals.len() / 3;
    let num_leaves = n >> group_log;
    assert!(num_leaves << group_log == n, "whole groups only");
    let be = backend()?;
    let stream = be.next_stream();
    let evals_dev = stream.clone_htod(evals)?;
    let nodes_dev =
        commit_group_leaves(&stream, be, hash, &evals_dev, num_leaves, 1u64 << group_log)?;
    let out = stream.clone_dtoh(&nodes_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Gather interleaved ext3 elements at `positions` from a resident evals
/// buffer — a small D2H of only the queried values (the FRI query phase's
/// `evaluation[index ^ 1]` reads).
pub fn gather_ext3_at(
    evals: &CudaSlice<u64>,
    positions: &[u32],
    stream: &Arc<CudaStream>,
) -> Result<Vec<u64>> {
    let q = positions.len();
    if q == 0 {
        return Ok(Vec::new());
    }
    // Guard the kernel's device reads: a position past the evals buffer would
    // be a silent out-of-bounds read. Positions are valid by construction;
    // this catches a caller bug host-side before it becomes device garbage
    // (matching `gather_merkle_paths_dev`).
    assert!(
        positions.iter().all(|&p| (p as usize) < evals.len() / 3),
        "gather_ext3_at: position >= evals length"
    );
    let be = backend()?;
    let pos_dev = stream.clone_htod(positions)?;
    // SAFETY: the gather kernel writes all 3 * q slots.
    let mut out_dev = unsafe { stream.alloc::<u64>(3 * q) }?;
    let cfg = LaunchConfig {
        grid_dim: ((q as u32).div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let q_u64 = q as u64;
    unsafe {
        stream
            .launch_builder(&be.gather_ext3_at)
            .arg(evals)
            .arg(&pos_dev)
            .arg(&q_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}
