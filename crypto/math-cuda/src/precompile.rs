//! On-GPU precompile (ECALL) kernels — Phase 6 of the trace-gen effort.
//!
//! Starts with the KeccakPermute precompile: batch the Keccak-f[1600] permutation on device so a
//! KECCAK ecall can compute its 25-lane output (and thus its memory effects) without the host
//! `keccak_f1600`. Reuses the validated `keccak_f1600` device function the Merkle tree already uses.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// Number of columns in the COMMIT trace table (mirrors `commit::cols::NUM_COLUMNS`).
pub const COMMIT_NCOLS: usize = 19;
/// Packed per-op stride the `commit_fill` kernel reads: [ts, index, address, count, first, end, value].
pub const COMMIT_STRIDE: usize = 7;

/// Fill the COMMIT (ECALL) trace table on device from the packed CommitOperation SoA, returning
/// the device buffer (`num_rows * COMMIT_NCOLS`, row-major) + `num_rows`. `ops_flat` is
/// `n * COMMIT_STRIDE`. Bit-identical to `commit::generate_commit_trace`. Moves the per-byte fill
/// to the GPU (the op-collection — memory reads — stays on host).
pub fn gpu_build_commit_trace_dev(ops_flat: &[u64], n: usize) -> Result<(CudaSlice<u64>, usize)> {
    assert_eq!(ops_flat.len(), n * COMMIT_STRIDE, "commit ops SoA length");
    let be = backend()?;
    let stream = be.next_stream();
    let num_rows = n.next_power_of_two().max(4);
    let ops_d = stream.clone_htod(ops_flat)?;
    let mut out = stream.alloc_zeros::<u64>(num_rows * COMMIT_NCOLS)?;
    let (n_u64, nr_u64) = (n as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.commit_fill)
            .arg(&ops_d)
            .arg(&n_u64)
            .arg(&nr_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Vec-returning parity API for [`gpu_build_commit_trace_dev`].
pub fn gpu_build_commit_trace(ops_flat: &[u64], n: usize) -> Result<Vec<u64>> {
    let (buf, _) = gpu_build_commit_trace_dev(ops_flat, n)?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// Number of columns in the main KECCAK (permute) trace table (mirrors `keccak::cols::NUM_COLUMNS`).
pub const KECCAK_TBL_NCOLS: usize = 511;
/// Packed per-op stride `keccak_table_fill` reads: [ts, state_addr, input[25], output[25]].
pub const KECCAK_TBL_STRIDE: usize = 52;

/// Fill the main KECCAK (permute) trace table on device from the packed KeccakOperation SoA,
/// returning `(device buffer, num_rows)`. `ops_flat` is `n * KECCAK_TBL_STRIDE`. Bit-identical to
/// `keccak::generate_keccak_trace`. The keccak COMPUTATION (input→output) is done elsewhere
/// (host `keccak_f1600` or `gpu_keccak_f1600_batch`); this fills the trace bytes.
pub fn gpu_build_keccak_trace_dev(ops_flat: &[u64], n: usize) -> Result<(CudaSlice<u64>, usize)> {
    assert_eq!(ops_flat.len(), n * KECCAK_TBL_STRIDE, "keccak ops SoA length");
    let be = backend()?;
    let stream = be.next_stream();
    let num_rows = n.next_power_of_two().max(4);
    let ops_d = stream.clone_htod(ops_flat)?;
    let mut out = stream.alloc_zeros::<u64>(num_rows * KECCAK_TBL_NCOLS)?;
    let (n_u64, nr_u64) = (n as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.keccak_table_fill)
            .arg(&ops_d)
            .arg(&n_u64)
            .arg(&nr_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Vec-returning parity API for [`gpu_build_keccak_trace_dev`].
pub fn gpu_build_keccak_trace(ops_flat: &[u64], n: usize) -> Result<Vec<u64>> {
    let (buf, _) = gpu_build_keccak_trace_dev(ops_flat, n)?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// Columns / packed strides for the ECDAS table device fill (mirror `ecdas::cols::NUM_COLUMNS`).
pub const ECDAS_NCOLS: usize = 521;
/// Per-op byte layout: x_g[32] y_g[32] x_a[32] y_a[32] round op x_r[32] y_r[32] lambda[32]
/// q0[33] q1[33] q2[33] next_op (column order).
pub const ECDAS_BSTRIDE: usize = 326;
/// Per-op signed carries: c0[64] c1[64] c2[64].
pub const ECDAS_CSTRIDE: usize = 192;

/// Fill the ECDAS trace table on device — pure FORMATTING of the precomputed witness (no EC math).
/// `bytes`=`n*ECDAS_BSTRIDE`, `carries`=`n*ECDAS_CSTRIDE` (i64), `ts`=`n`. Bit-identical to
/// `ecdas::generate_ecdas_trace`. Returns `(device buffer, num_rows)`.
pub fn gpu_build_ecdas_trace_dev(
    bytes: &[u8],
    carries: &[i64],
    ts: &[u64],
    n: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    assert_eq!(bytes.len(), n * ECDAS_BSTRIDE, "ecdas bytes length");
    assert_eq!(carries.len(), n * ECDAS_CSTRIDE, "ecdas carries length");
    assert_eq!(ts.len(), n, "ecdas ts length");
    let be = backend()?;
    let stream = be.next_stream();
    let num_rows = n.next_power_of_two().max(4);
    let bytes_d = stream.clone_htod(bytes)?;
    let carries_d = stream.clone_htod(carries)?;
    let ts_d = stream.clone_htod(ts)?;
    let mut out = stream.alloc_zeros::<u64>(num_rows * ECDAS_NCOLS)?;
    let (n_u64, nr_u64) = (n as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.ecdas_fill)
            .arg(&bytes_d)
            .arg(&carries_d)
            .arg(&ts_d)
            .arg(&n_u64)
            .arg(&nr_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Vec-returning parity API for [`gpu_build_ecdas_trace_dev`].
pub fn gpu_build_ecdas_trace(bytes: &[u8], carries: &[i64], ts: &[u64], n: usize) -> Result<Vec<u64>> {
    let (buf, _) = gpu_build_ecdas_trace_dev(bytes, carries, ts, n)?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// Compute the ECDAS per-step CARRIES on device — the `conv` limb-convolution witness that the CPU
/// `ecsm::witness::build_step` computed (~190ms/proof for ethrex_5tx). `bytes`=`n*ECDAS_BSTRIDE`
/// (same packing as the ecdas fill: the point + quotient limbs the carries are derived from). Returns
/// the `n*ECDAS_CSTRIDE` signed carries (c0[64] c1[64] c2[64] per step) resident on device, bit-exact
/// with the CPU `carries_lambda/xr/yr` + `limb_carries`. The EC scalar-mult + quotients stay on CPU.
pub fn gpu_build_ecdas_carries_dev(bytes: &[u8], n: usize) -> Result<CudaSlice<i64>> {
    assert_eq!(bytes.len(), n * ECDAS_BSTRIDE, "ecdas carries: bytes length");
    let be = backend()?;
    let stream = be.next_stream();
    let bytes_d = stream.clone_htod(bytes)?;
    let mut out = stream.alloc_zeros::<i64>(n.max(1) * ECDAS_CSTRIDE)?;
    if n > 0 {
        let n_u64 = n as u64;
        // One thread per step; the kernel is register/local-mem-heavy (10 limb arrays + __int128
        // convolutions), so use a small block (like keccak_rnd_fill) to avoid LAUNCH_OUT_OF_RESOURCES.
        let block = 32u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.ecdas_carries)
                .arg(&bytes_d)
                .arg(&n_u64)
                .arg(&mut out)
                .launch(cfg)?;
        }
    }
    stream.synchronize()?;
    Ok(out)
}

/// Vec-returning parity API for [`gpu_build_ecdas_carries_dev`] (returns `n*ECDAS_CSTRIDE` carries).
pub fn gpu_build_ecdas_carries(bytes: &[u8], n: usize) -> Result<Vec<i64>> {
    let buf = gpu_build_ecdas_carries_dev(bytes, n)?;
    let be = backend()?;
    let stream = be.next_stream();
    let mut host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    host.truncate(n * ECDAS_CSTRIDE);
    Ok(host)
}

/// Columns / packed strides for the ECSM table device fill (mirror `ecsm::cols::NUM_COLUMNS`).
pub const ECSM_NCOLS: usize = 667;
/// Per-op byte layout: x_r[32] y_r[32] k[32] x_g[32] y_g[32] x2[32] q0[32] q1[33]
/// x_g_sub_p[32] k_sub_n[32] x_r_sub_p[32] len_k.
pub const ECSM_BSTRIDE: usize = 354;
/// Per-op signed carries: c0[64] c1[64].
pub const ECSM_CSTRIDE: usize = 128;
/// Per-op addresses: ts, addr_xg, addr_k, addr_xr.
pub const ECSM_ASTRIDE: usize = 4;

/// Fill the ECSM trace table on device — pure FORMATTING of the precomputed witness (no EC math).
/// `bytes`=`n*ECSM_BSTRIDE`, `carries`=`n*ECSM_CSTRIDE` (i64), `addrs`=`n*ECSM_ASTRIDE`.
/// Bit-identical to `ecsm::generate_ecsm_trace`. Returns `(device buffer, num_rows)`.
pub fn gpu_build_ecsm_trace_dev(
    bytes: &[u8],
    carries: &[i64],
    addrs: &[u64],
    n: usize,
) -> Result<(CudaSlice<u64>, usize)> {
    assert_eq!(bytes.len(), n * ECSM_BSTRIDE, "ecsm bytes length");
    assert_eq!(carries.len(), n * ECSM_CSTRIDE, "ecsm carries length");
    assert_eq!(addrs.len(), n * ECSM_ASTRIDE, "ecsm addrs length");
    let be = backend()?;
    let stream = be.next_stream();
    let num_rows = n.next_power_of_two().max(4);
    let bytes_d = stream.clone_htod(bytes)?;
    let carries_d = stream.clone_htod(carries)?;
    let addrs_d = stream.clone_htod(addrs)?;
    let mut out = stream.alloc_zeros::<u64>(num_rows * ECSM_NCOLS)?;
    let (n_u64, nr_u64) = (n as u64, num_rows as u64);
    unsafe {
        stream
            .launch_builder(&be.ecsm_fill)
            .arg(&bytes_d)
            .arg(&carries_d)
            .arg(&addrs_d)
            .arg(&n_u64)
            .arg(&nr_u64)
            .arg(&mut out)
            .launch(LaunchConfig::for_num_elems(num_rows as u32))?;
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Vec-returning parity API for [`gpu_build_ecsm_trace_dev`].
pub fn gpu_build_ecsm_trace(bytes: &[u8], carries: &[i64], addrs: &[u64], n: usize) -> Result<Vec<u64>> {
    let (buf, _) = gpu_build_ecsm_trace_dev(bytes, carries, addrs, n)?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// Columns for the KECCAK_RND (per-round) table; 24 rows per op (mirror `keccak_rnd::cols`).
pub const KECCAK_RND_NCOLS: usize = 1480;
/// Per-op packed stride the `keccak_rnd_fill` kernel reads: [timestamp, input[25]].
pub const KECCAK_RND_STRIDE: usize = 26;

/// Fill the KECCAK_RND (per-round) trace table on device — recomputes the 24 permutation rounds
/// per op (theta/rho/pi/chi/iota with byte + HWSL-carry decompositions) and writes all 1480 cols
/// per round. One thread per op (state evolves sequentially). `ops_flat` is `n * KECCAK_RND_STRIDE`.
/// Bit-identical to `keccak_rnd::generate_keccak_rnd_trace`. Register-heavy → small block.
pub fn gpu_build_keccak_rnd_trace_dev(ops_flat: &[u64], n: usize) -> Result<(CudaSlice<u64>, usize)> {
    assert_eq!(ops_flat.len(), n * KECCAK_RND_STRIDE, "keccak_rnd ops length");
    let be = backend()?;
    let stream = be.next_stream();
    let num_rows = (n * 24).next_power_of_two().max(4);
    let ops_d = stream.clone_htod(ops_flat)?;
    let mut out = stream.alloc_zeros::<u64>(num_rows * KECCAK_RND_NCOLS)?;
    let n_u64 = n as u64;
    let nr_u64 = num_rows as u64;
    if n > 0 {
        // One thread per op; the round kernel is register-heavy, so use a small block.
        let block = 32u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&be.keccak_rnd_fill)
                .arg(&ops_d)
                .arg(&n_u64)
                .arg(&nr_u64)
                .arg(&mut out)
                .launch(cfg)?;
        }
    }
    stream.synchronize()?;
    Ok((out, num_rows))
}

/// Vec-returning parity API for [`gpu_build_keccak_rnd_trace_dev`].
pub fn gpu_build_keccak_rnd_trace(ops_flat: &[u64], n: usize) -> Result<Vec<u64>> {
    let (buf, _) = gpu_build_keccak_rnd_trace_dev(ops_flat, n)?;
    let be = backend()?;
    let stream = be.next_stream();
    let host = stream.clone_dtoh(&buf)?;
    stream.synchronize()?;
    Ok(host)
}

/// Apply Keccak-f[1600] to a batch of 25-lane states on device, in place. `flat.len()` must be a
/// multiple of 25; state `i` is `flat[i*25 .. i*25+25]`. Returns the permuted flat buffer.
/// Bit-identical to `executor::vm::instruction::execution::keccak_f1600` applied per state.
pub fn gpu_keccak_f1600_batch(flat: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(flat.len() % 25, 0, "keccak state batch must be a multiple of 25 lanes");
    let n = flat.len() / 25;
    let be = backend()?;
    let stream = be.next_stream();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut states = stream.clone_htod(flat)?;
    let n_u64 = n as u64;
    // keccak_f1600 is register-heavy (~60 u64 locals), so a large block overflows the register
    // file (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES). Use a small block.
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.keccak_f1600_batch)
            .arg(&n_u64)
            .arg(&mut states)
            .launch(cfg)?;
    }
    let host = stream.clone_dtoh(&states)?;
    stream.synchronize()?;
    Ok(host)
}
