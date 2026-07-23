//! Device `CpuOperation` builder (Phase 0 of full-GPU trace-gen; see
//! `kernels/trace_ops.cu`).
//!
//! Reconstructs, on device, the per-cycle op record the prover's
//! `CpuOperation::from_log` computes on the host — a stateless function of the cycle's
//! executor Log + decoded instruction. This SoA is the resident seam every later
//! trace-gen stage reads. `gpu_build_cpu_ops` is the host-in/host-out form used by the
//! byte-parity test; a device-resident form follows once downstream stages consume it.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// The computed per-cycle fields (indexed by cycle). `flags` bit 0 = branch_cond,
/// 1 = ecall_commit, 2 = ecall_keccak, 3 = ecall_ecsm. The decode (pc/imm/packed),
/// timestamp, and raw log values are inputs, so only the computed outputs are returned.
pub struct DeviceCpuOps {
    pub rv1: Vec<u64>,
    pub rv2: Vec<u64>,
    pub arg2: Vec<u64>,
    pub res: Vec<u64>,
    pub rvd: Vec<u64>,
    pub next_pc: Vec<u64>,
    pub flags: Vec<u8>,
    pub commit_buf_addr: Vec<u64>,
    pub commit_count: Vec<u64>,
    pub keccak_state_addr: Vec<u64>,
}

/// Build the `CpuOperation` computed fields on device from the per-cycle Log SoA
/// (`current_pc`, `next_pc_log`, `src1_val`, `src2_val`, `dst_val`) and decode SoA
/// (`pc`, `imm`, `packed` = `ShrunkDecode::pack()`). One thread per cycle.
#[allow(clippy::too_many_arguments)]
pub fn gpu_build_cpu_ops(
    current_pc: &[u64],
    next_pc_log: &[u64],
    src1_val: &[u64],
    src2_val: &[u64],
    dst_val: &[u64],
    pc: &[u64],
    imm: &[u64],
    packed: &[u64],
) -> Result<DeviceCpuOps> {
    let be = backend()?;
    let stream = be.next_stream();
    let n = current_pc.len();
    if n == 0 {
        return Ok(DeviceCpuOps {
            rv1: vec![],
            rv2: vec![],
            arg2: vec![],
            res: vec![],
            rvd: vec![],
            next_pc: vec![],
            flags: vec![],
            commit_buf_addr: vec![],
            commit_count: vec![],
            keccak_state_addr: vec![],
        });
    }
    debug_assert_eq!(next_pc_log.len(), n);
    debug_assert_eq!(src1_val.len(), n);
    debug_assert_eq!(src2_val.len(), n);
    debug_assert_eq!(dst_val.len(), n);
    debug_assert_eq!(pc.len(), n);
    debug_assert_eq!(imm.len(), n);
    debug_assert_eq!(packed.len(), n);

    let cpc_d = stream.clone_htod(current_pc)?;
    let npc_d = stream.clone_htod(next_pc_log)?;
    let s1_d = stream.clone_htod(src1_val)?;
    let s2_d = stream.clone_htod(src2_val)?;
    let dv_d = stream.clone_htod(dst_val)?;
    let pc_d = stream.clone_htod(pc)?;
    let imm_d = stream.clone_htod(imm)?;
    let pk_d = stream.clone_htod(packed)?;

    let mut rv1 = stream.alloc_zeros::<u64>(n)?;
    let mut rv2 = stream.alloc_zeros::<u64>(n)?;
    let mut arg2 = stream.alloc_zeros::<u64>(n)?;
    let mut res = stream.alloc_zeros::<u64>(n)?;
    let mut rvd = stream.alloc_zeros::<u64>(n)?;
    let mut next_pc = stream.alloc_zeros::<u64>(n)?;
    let mut flags = stream.alloc_zeros::<u8>(n)?;
    let mut commit_buf_addr = stream.alloc_zeros::<u64>(n)?;
    let mut commit_count = stream.alloc_zeros::<u64>(n)?;
    let mut keccak_state_addr = stream.alloc_zeros::<u64>(n)?;

    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.build_cpu_ops)
            .arg(&n_u64)
            .arg(&cpc_d)
            .arg(&npc_d)
            .arg(&s1_d)
            .arg(&s2_d)
            .arg(&dv_d)
            .arg(&pc_d)
            .arg(&imm_d)
            .arg(&pk_d)
            .arg(&mut rv1)
            .arg(&mut rv2)
            .arg(&mut arg2)
            .arg(&mut res)
            .arg(&mut rvd)
            .arg(&mut next_pc)
            .arg(&mut flags)
            .arg(&mut commit_buf_addr)
            .arg(&mut commit_count)
            .arg(&mut keccak_state_addr)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    Ok(DeviceCpuOps {
        rv1: stream.clone_dtoh(&rv1)?,
        rv2: stream.clone_dtoh(&rv2)?,
        arg2: stream.clone_dtoh(&arg2)?,
        res: stream.clone_dtoh(&res)?,
        rvd: stream.clone_dtoh(&rvd)?,
        next_pc: stream.clone_dtoh(&next_pc)?,
        flags: stream.clone_dtoh(&flags)?,
        commit_buf_addr: stream.clone_dtoh(&commit_buf_addr)?,
        commit_count: stream.clone_dtoh(&commit_count)?,
        keccak_state_addr: stream.clone_dtoh(&keccak_state_addr)?,
    })
}
