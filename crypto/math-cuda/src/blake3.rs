//! GPU BLAKE3 for Merkle commits.
//!
//! Twin of [`crate::merkle`]'s keccak path, kernel for kernel, so the two read
//! against each other. Keccak stays the prover's default hash: nothing in the
//! production dispatch reaches this module yet.
//!
//! What is here so far is the compression function and the oracle that checks
//! it. The device compression is not otherwise reachable from host code, so
//! without [`compress_probe`] there would be nothing to check it against the
//! host reference with.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
/// Threads per block for the BLAKE3 kernels.
///
/// Wider than [`crate::merkle`]'s 128 because the register footprint is a third
/// of keccak's: 16 working-state words + 16 message words + the output, all u32,
/// against keccak's 25 u64 lanes plus a 25-lane scratch. The 128 there is a
/// Blackwell register-file limit, not a shape this path shares.
const BLAKE3_BLOCK_DIM: u32 = 256;

pub(crate) fn blake3_launch_cfg(num_threads: u64) -> LaunchConfig {
    debug_assert!(
        num_threads <= u32::MAX as u64,
        "blake3_launch_cfg: num_threads ({num_threads}) exceeds u32 grid range",
    );
    let grid = (num_threads as u32).div_ceil(BLAKE3_BLOCK_DIM);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLAKE3_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One compression's inputs, in the argument order of the host reference
/// `blake3_compress_rounds(h, m, t, block_len, flags, rounds)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressInput {
    pub h: [u32; 8],
    pub m: [u32; 16],
    pub t: u64,
    pub block_len: u32,
    pub flags: u32,
}

/// Which round count [`compress_probe`] should run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeRounds {
    /// 6 — the internal variant.
    Six,
    /// 7 — standard BLAKE3, where the `blake3` crate is an external anchor.
    Seven,
    /// Whatever this cubin is compiled for — the round count the production
    /// kernels built from `blake3_compress` will use.
    CompiledIn,
}

/// Parity harness: run the device compression function over `inputs` and return
/// each full 16-word output.
///
/// Not a production path — the device compression is otherwise unreachable from
/// host code, so without this there would be nothing to check it against the
/// host reference with.
pub fn compress_probe(inputs: &[CompressInput], rounds: ProbeRounds) -> Result<Vec<[u32; 16]>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let n = inputs.len();
    let mut h = Vec::with_capacity(n * 8);
    let mut m = Vec::with_capacity(n * 16);
    let mut t = Vec::with_capacity(n);
    let mut block_len = Vec::with_capacity(n);
    let mut flags = Vec::with_capacity(n);
    for i in inputs {
        h.extend_from_slice(&i.h);
        m.extend_from_slice(&i.m);
        t.push(i.t);
        block_len.push(i.block_len);
        flags.push(i.flags);
    }

    let be = backend()?;
    let stream = be.next_stream();
    let h_dev = stream.clone_htod(&h)?;
    let m_dev = stream.clone_htod(&m)?;
    let t_dev = stream.clone_htod(&t)?;
    let bl_dev = stream.clone_htod(&block_len)?;
    let fl_dev = stream.clone_htod(&flags)?;
    let mut out_dev = stream.alloc_zeros::<u32>(n * 16)?;

    let kernel = match rounds {
        ProbeRounds::Six => &be.blake3_compress_probe_6r,
        ProbeRounds::Seven => &be.blake3_compress_probe_7r,
        ProbeRounds::CompiledIn => &be.blake3_compress_probe_default,
    };
    let n_u64 = n as u64;
    let cfg = blake3_launch_cfg(n_u64);
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&h_dev)
            .arg(&m_dev)
            .arg(&t_dev)
            .arg(&bl_dev)
            .arg(&fl_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let flat = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(flat
        .chunks_exact(16)
        .map(|c| {
            let mut w = [0u32; 16];
            w.copy_from_slice(c);
            w
        })
        .collect())
}

/// The round count `kernels/blake3.cu` was compiled for.
///
/// The host tree's round count and this one are separate crates' features, so
/// nothing forces them equal; a mismatch would be a GPU tree committing under a
/// different hash than the CPU one, with no symptom short of a failing verify.
/// Reading it back makes that assertable.
pub fn device_rounds() -> Result<u32> {
    let be = backend()?;
    let stream = be.next_stream();
    let mut out_dev = stream.alloc_zeros::<u32>(1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.blake3_rounds_probe)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out[0])
}
