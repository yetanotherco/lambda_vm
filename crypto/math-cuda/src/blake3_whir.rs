//! The WHIR base proof's BLAKE3 kernels (D-B3HYB S3): the
//! [`crate::DeviceHash::Blake3`] arm of the WHIR leaf, tree and grind
//! dispatches, at **7 rounds** — standard BLAKE3, the host's
//! `crypto::hash::blake3::Blake3Std` — whatever round count the STARK
//! pipeline's `blake3.cubin` is built at (math-cuda's `blake3-6round`).
//!
//! The kernels are `kernels/blake3_whir.cu`, which instantiates `blake3.cu` a
//! second time at 7 rounds, loaded on first use from that cubin, never by
//! [`crate::device::Backend::init`]. The loader reads the module's round count
//! back and refuses a module that is not 7.
//!
//! Only the WHIR paths reach here (`crate::whir`, `crate::whir_coset`,
//! `crate::grinding`): the STARK pipeline's `DeviceHash::Blake3` dispatches
//! keep `crate::blake3`'s handles at the build's round count.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::Result;
use crate::device::backend;

const BLAKE3_WHIR_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/blake3_whir.cubin"));

/// The round count every kernel here is compiled for.
pub const ROUNDS: u32 = 7;

/// Threads per block: `crate::blake3`'s width, for the same register reason.
const BLOCK_DIM: u32 = 256;

/// The kernels.
pub struct Kernels {
    _module: Arc<CudaModule>,
    /// Coset leaves over a base / ext3 codeword (`crate::whir`).
    pub leaves_base_coset: CudaFunction,
    /// [`Self::leaves_base_coset`] with block `j` at position `bitrev(j)`.
    pub leaves_base_coset_rev: CudaFunction,
    pub leaves_ext3_coset: CudaFunction,
    /// The joint leaves of a grouped run's 2–4 base codewords
    /// (`crate::whir::launch_joint_leaves`).
    pub leaves_base_coset_joint: CudaFunction,
    /// The joint leaf's prefix midstate, for the host to check
    /// ([`joint_midstate`]).
    pub joint_midstate_probe: CudaFunction,
    pub merkle_level: CudaFunction,
    pub merkle_tail: CudaFunction,
    pub grind_search: CudaFunction,
}

/// Load the kernels once, refusing a module whose round count is not
/// [`ROUNDS`].
pub fn kernels() -> Result<&'static Kernels> {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    if let Some(k) = KERNELS.get() {
        return Ok(k);
    }
    let be = backend()?;
    let module = be
        .ctx
        .load_module(Ptx::from_binary(BLAKE3_WHIR_CUBIN.to_vec()))?;
    let rounds = {
        let probe = module.load_function("b3w_rounds_probe")?;
        let stream = be.next_stream();
        let mut out = stream.alloc_zeros::<u32>(1)?;
        unsafe {
            stream
                .launch_builder(&probe)
                .arg(&mut out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let host = crate::readback::clone_dtoh(&stream, &out)?;
        stream.synchronize()?;
        host[0]
    };
    if rounds != ROUNDS {
        return Err(crate::invalid_value());
    }
    let k = Kernels {
        leaves_base_coset: module.load_function("b3w_leaves_base_coset")?,
        leaves_base_coset_rev: module.load_function("b3w_leaves_base_coset_rev")?,
        leaves_ext3_coset: module.load_function("b3w_leaves_ext3_coset")?,
        leaves_base_coset_joint: module.load_function("b3w_leaves_base_coset_joint")?,
        joint_midstate_probe: module.load_function("b3w_joint_midstate_probe")?,
        merkle_level: module.load_function("blake3_merkle_level")?,
        merkle_tail: module.load_function("blake3_merkle_tail")?,
        grind_search: module.load_function("b3w_grind_search")?,
        _module: module,
    };
    Ok(KERNELS.get_or_init(|| k))
}

/// The compiled round count of the loaded module (7, or the load fails).
pub fn device_rounds() -> Result<u32> {
    kernels().map(|_| ROUNDS)
}

pub(crate) fn launch_cfg(threads: u64) -> LaunchConfig {
    debug_assert!(threads <= u32::MAX as u64, "blake3 whir launch: {threads}");
    LaunchConfig {
        grid_dim: ((threads as u32).div_ceil(BLOCK_DIM).max(1), 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Fill the inner nodes of a binary tree whose `leaves_len` leaves are in the
/// tail of `nodes_dev` (the host layout: root at node 0), at 7 rounds.
/// `crate::blake3::build_inner_tree_levels`' walk on this module's handles.
pub(crate) fn build_inner_tree_levels(
    stream: &CudaStream,
    nodes_dev: &mut CudaSlice<u8>,
    leaves_len: usize,
) -> Result<()> {
    let k = kernels()?;
    const TAIL_MAX_PAIRS: u64 = BLOCK_DIM as u64;
    let mut level_begin: u64 = (leaves_len - 1) as u64;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        if n_pairs <= TAIL_MAX_PAIRS {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&k.merkle_tail)
                    .arg(&mut *nodes_dev)
                    .arg(&level_begin)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        unsafe {
            stream
                .launch_builder(&k.merkle_level)
                .arg(&mut *nodes_dev)
                .arg(&new_begin)
                .arg(&n_pairs)
                .launch(launch_cfg(n_pairs))?;
        }
        level_begin = new_begin;
    }
    Ok(())
}

/// The chaining value the joint-leaf kernel starts every leaf's chain from,
/// for `blocks` blocks of `2^log_block` base values: what the host's
/// `crypto::merkle_tree::backends::blake3_joint::joint_leaf_midstate` gives.
pub fn joint_midstate(blocks: u64, log_block: u32) -> Result<[u32; 8]> {
    let k = kernels()?;
    let stream = backend()?.next_stream();
    let mut out = stream.alloc_zeros::<u32>(8)?;
    unsafe {
        stream
            .launch_builder(&k.joint_midstate_probe)
            .arg(&blocks)
            .arg(&log_block)
            .arg(&mut out)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let host = crate::readback::clone_dtoh(&stream, &out)?;
    stream.synchronize()?;
    let mut cv = [0u32; 8];
    cv.copy_from_slice(&host);
    Ok(cv)
}

/// The grind kernel and its block size, for [`crate::grinding`]'s range walk.
pub(crate) fn grind_kernel() -> Result<(&'static CudaFunction, u32)> {
    Ok((&kernels()?.grind_search, BLOCK_DIM))
}
